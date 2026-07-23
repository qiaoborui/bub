from __future__ import annotations

import asyncio
import contextlib
import json
import re
import secrets
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from html import escape
from html.parser import HTMLParser
from typing import Any, ClassVar

from loguru import logger
from markdown_it import MarkdownIt
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from telegram import Bot, Message, Update
from telegram.constants import MessageLimit, ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, filters
from telegram.ext import MessageHandler as TelegramMessageHandler
from telegram.request import HTTPXRequest

from bub import config
from bub.channels.base import Channel
from bub.channels.contracts import MessageHandler
from bub.channels.message import ChannelMessage, MediaItem, MediaType
from bub.configure import Settings, ensure_config
from bub.streaming import StreamEvent
from bub.utils import exclude_none


@config(name="telegram")
class TelegramSettings(Settings):
    model_config = SettingsConfigDict(env_prefix="BUB_TELEGRAM_", extra="ignore", env_file=".env")

    token: str = Field(default="", description="Telegram bot token.")
    allow_users: str | None = Field(
        default=None, description="Comma-separated list of allowed Telegram user IDs, or empty for no restriction."
    )
    allow_chats: str | None = Field(
        default=None, description="Comma-separated list of allowed Telegram chat IDs, or empty for no restriction."
    )
    proxy: str | None = Field(
        default=None,
        description="Optional proxy URL for connecting to Telegram API, e.g. 'http://user:pass@host:port' or 'socks5://host:port'.",
    )
    stream_update_interval: float = Field(
        default=0.5,
        ge=0.1,
        description="Minimum seconds between Telegram sendMessageDraft updates.",
    )


NO_ACCESS_MESSAGE = "You are not allowed to chat with me. Please deploy your own instance of Bub."
TELEGRAM_TEXT_LIMIT = int(MessageLimit.MAX_TEXT_LENGTH)
_MARKDOWN = MarkdownIt("commonmark", {"breaks": True, "html": False}).enable("strikethrough")
_PARSE_ERROR_MARKERS = ("parse entities", "unsupported start tag", "can't find end tag")


class _TelegramHTMLParser(HTMLParser):
    _INLINE_TAGS: ClassVar[dict[str, str]] = {
        "strong": "b",
        "b": "b",
        "em": "i",
        "i": "i",
        "s": "s",
        "strike": "s",
        "del": "s",
        "u": "u",
        "ins": "u",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._lists: list[list[str | int]] = []
        self._pre_depth = 0

    def _append_newline(self, count: int = 1) -> None:
        current = "".join(self._parts[-2:])
        missing = count - len(current) + len(current.rstrip("\n"))
        if missing > 0:
            self._parts.append("\n" * missing)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._handle_inline_start(tag, attrs):
            return
        self._handle_block_start(tag, attrs)

    def _handle_inline_start(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if mapped := self._INLINE_TAGS.get(tag):
            self._parts.append(f"<{mapped}>")
            return True
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            self._parts.append(f'<a href="{escape(href, quote=True)}">')
            return True
        elif tag == "pre":
            self._append_newline()
            self._parts.append("<pre>")
            self._pre_depth += 1
            return True
        elif tag == "code":
            language = dict(attrs).get("class") or ""
            if language.startswith("language-"):
                self._parts.append(f'<code class="{escape(language, quote=True)}">')
            else:
                self._parts.append("<code>")
            return True
        return False

    def _handle_block_start(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._append_newline()
            self._parts.append("<b>")
        elif tag == "blockquote":
            self._append_newline()
            self._parts.append("<blockquote>")
        elif tag == "br":
            self._parts.append("\n")
        elif tag == "hr":
            self._append_newline()
            self._parts.append("———\n")
        elif tag in {"ul", "ol"}:
            start = int(dict(attrs).get("start") or 1)
            self._lists.append([tag, start - 1])
        elif tag == "li":
            self._append_newline()
            prefix = "• "
            if self._lists and self._lists[-1][0] == "ol":
                self._lists[-1][1] = int(self._lists[-1][1]) + 1
                prefix = f"{self._lists[-1][1]}. "
            self._parts.append("  " * max(0, len(self._lists) - 1) + prefix)
        elif tag == "img":
            attributes = dict(attrs)
            alt = attributes.get("alt") or "image"
            src = attributes.get("src") or ""
            self._parts.append(escape(f"{alt} ({src})" if src else alt))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if mapped := self._INLINE_TAGS.get(tag):
            self._parts.append(f"</{mapped}>")
        elif tag in {"a", "code", "pre", "blockquote"}:
            self._parts.append(f"</{tag}>")
            if tag == "pre":
                self._pre_depth -= 1
            if tag in {"pre", "blockquote"}:
                self._append_newline(2)
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("</b>")
            self._append_newline(2)
        elif tag == "p":
            self._append_newline(2)
        elif tag == "li":
            self._append_newline()
        elif tag in {"ul", "ol"}:
            if self._lists:
                self._lists.pop()
            self._append_newline()

    def handle_data(self, data: str) -> None:
        if self._pre_depth == 0 and "\n" in data and not data.strip():
            return
        self._parts.append(escape(data))

    def rendered(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def _draft_preview(text: str) -> str:
    if len(text) <= TELEGRAM_TEXT_LIMIT:
        return text
    return "…" + text[-(TELEGRAM_TEXT_LIMIT - 1) :]


def _split_text(text: str) -> list[str]:
    return [text[offset : offset + TELEGRAM_TEXT_LIMIT] for offset in range(0, len(text), TELEGRAM_TEXT_LIMIT)]


def _telegram_text(content: str) -> str:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content
    if not isinstance(data, dict):
        return content
    message = data.get("message", "")
    return message if isinstance(message, str) else str(message)


def _telegram_html(text: str) -> str:
    parser = _TelegramHTMLParser()
    parser.feed(_MARKDOWN.render(text))
    parser.close()
    return parser.rendered()


def _is_parse_error(error: BadRequest) -> bool:
    detail = str(error).lower()
    return any(marker in detail for marker in _PARSE_ERROR_MARKERS)


async def _render_draft_preview(message: ChannelMessage, value: str) -> tuple[str, str] | None:
    rendered = await message.stream_renderer(value) if message.stream_renderer is not None else value
    if rendered is None:
        return None
    preview = _draft_preview(_telegram_text(rendered))
    return (preview, _telegram_html(preview)) if preview.strip() else None


def _message_type(message: Message) -> str:
    if getattr(message, "text", None):
        return "text"
    if getattr(message, "photo", None):
        return "photo"
    if getattr(message, "audio", None):
        return "audio"
    if getattr(message, "sticker", None):
        return "sticker"
    if getattr(message, "video", None):
        return "video"
    if getattr(message, "voice", None):
        return "voice"
    if getattr(message, "document", None):
        return "document"
    if getattr(message, "video_note", None):
        return "video_note"
    return "unknown"


class BubMessageFilter(filters.MessageFilter):
    GROUP_CHAT_TYPES: ClassVar[set[str]] = {"group", "supergroup"}

    def _content(self, message: Message) -> str:
        return (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()

    def filter(self, message: Message) -> bool | dict[str, list[Any]] | None:
        msg_type = _message_type(message)
        if msg_type == "unknown":
            return False

        # Private chat: process all non-command messages and bot commands.
        if message.chat.type == "private":
            return True

        # Group chat: only process when explicitly addressed to the bot.
        if message.chat.type in self.GROUP_CHAT_TYPES:
            bot = message.get_bot()
            bot_id = bot.id
            bot_username = (bot.username or "").lower()

            mentions_bot = self._mentions_bot(message, bot_id, bot_username)
            reply_to_bot = self._is_reply_to_bot(message, bot_id)

            if msg_type != "text" and not getattr(message, "caption", None):
                return reply_to_bot

            return mentions_bot or reply_to_bot

        return False

    def _mentions_bot(self, message: Message, bot_id: int, bot_username: str) -> bool:
        content = self._content(message).lower()
        mentions_by_keyword = "bub" in content or bool(bot_username and f"@{bot_username}" in content)

        entities = [*(getattr(message, "entities", None) or ()), *(getattr(message, "caption_entities", None) or ())]
        for entity in entities:
            if entity.type == "mention" and bot_username:
                mention_text = content[entity.offset : entity.offset + entity.length]
                if mention_text.lower() == f"@{bot_username}":
                    return True
                continue
            if entity.type == "text_mention" and entity.user and entity.user.id == bot_id:
                return True
        return mentions_by_keyword

    @staticmethod
    def _is_reply_to_bot(message: Message, bot_id: int) -> bool:
        reply_to_message = message.reply_to_message
        if reply_to_message is None or reply_to_message.from_user is None:
            return False
        return reply_to_message.from_user.id == bot_id


MESSAGE_FILTER = BubMessageFilter()

_MSG_TYPE_TO_MEDIA_TYPE: dict[str, MediaType] = {
    "photo": "image",
    "sticker": "image",
    "audio": "audio",
    "voice": "audio",
    "video": "video",
    "video_note": "video",
    "document": "document",
}


def _extract_media_items(metadata: dict[str, Any]) -> list[MediaItem]:
    """Extract MediaItem from parsed metadata, removing data_url from it."""
    media_dict = metadata.get("media")
    if not isinstance(media_dict, dict):
        return []
    data_fetcher = media_dict.pop("data_fetcher", None)
    if not data_fetcher or not callable(data_fetcher):
        return []
    msg_type = metadata.get("type", "")
    media_type = _MSG_TYPE_TO_MEDIA_TYPE.get(msg_type, "document")
    mime_type = media_dict.get("mime_type", "")
    return [
        MediaItem(type=media_type, data_fetcher=data_fetcher, mime_type=mime_type, filename=media_dict.get("file_name"))
    ]


class TelegramChannel(Channel):
    name = "telegram"
    _app: Application

    def __init__(self, on_receive: MessageHandler) -> None:
        self._on_receive = on_receive
        self._settings = ensure_config(TelegramSettings)
        self._allow_users = {uid.strip() for uid in (self._settings.allow_users or "").split(",") if uid.strip()}
        self._allow_chats = {cid.strip() for cid in (self._settings.allow_chats or "").split(",") if cid.strip()}
        self._parser = TelegramMessageParser(bot_getter=lambda: self._app.bot)
        self._typing_tasks: dict[str, asyncio.Task] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._settings.token)

    @property
    def needs_debounce(self) -> bool:
        return True

    def prepare_stream_output(self, message: ChannelMessage) -> bool:
        if (
            message.kind != "normal"
            or message.output_channel != "null"
            or message.context.get("chat_type") != "private"
        ):
            return False
        message.output_channel = self.name
        message.set_response_mode("direct_stream")
        return True

    async def start(self, stop_event: asyncio.Event) -> None:
        proxy = self._settings.proxy
        logger.info(
            "telegram.start allow_users_count={} allow_chats_count={} proxy_enabled={}",
            len(self._allow_users),
            len(self._allow_chats),
            bool(proxy),
        )
        get_updates_request = HTTPXRequest(read_timeout=30, proxy=proxy)
        builder = Application.builder().token(self._settings.token).get_updates_request(get_updates_request)
        if proxy:
            builder = builder.proxy(proxy)
        self._app = builder.build()
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("bub", self._on_message, has_args=True, block=False))
        self._app.add_handler(TelegramMessageHandler(~filters.COMMAND, self._on_message, block=False))
        await self._app.initialize()
        await self._app.start()
        updater = self._app.updater
        if updater is None:
            return
        await updater.start_polling(drop_pending_updates=True, allowed_updates=["message"])
        logger.info("telegram.start polling")

    async def stop(self) -> None:
        updater = self._app.updater
        with contextlib.suppress(Exception):
            if updater is not None and updater.running:
                await updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        for task in self._typing_tasks.values():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._typing_tasks.clear()
        logger.info("telegram.stopped")

    async def send(self, message: ChannelMessage) -> None:
        chat_id = message.chat_id
        text = _telegram_text(message.content)
        if not text.strip():
            return
        for chunk in _split_text(text):
            await self._send_text(chat_id, chunk)

    async def _send_text(self, chat_id: str, text: str) -> None:
        try:
            await self._app.bot.send_message(chat_id=chat_id, text=_telegram_html(text), parse_mode=ParseMode.HTML)
        except BadRequest as exc:
            if not _is_parse_error(exc):
                raise
            logger.warning("telegram.html fallback chat_id={} error={}", chat_id, exc)
            await self._app.bot.send_message(chat_id=chat_id, text=text)

    async def _send_draft(self, chat_id: int, draft_id: int, plain: str, html: str) -> None:
        try:
            await self._app.bot.send_message_draft(
                chat_id=chat_id,
                draft_id=draft_id,
                text=html,
                parse_mode=ParseMode.HTML,
            )
        except BadRequest as exc:
            if not _is_parse_error(exc):
                raise
            logger.warning("telegram.draft html fallback chat_id={} error={}", chat_id, exc)
            await self._app.bot.send_message_draft(chat_id=chat_id, draft_id=draft_id, text=plain)

    def stream_events(self, message: ChannelMessage, stream: AsyncIterable[StreamEvent]) -> AsyncIterable[StreamEvent]:
        if message.context.get("chat_type") != "private":
            return stream
        return self._stream_private_drafts(message, stream)

    async def _stream_private_drafts(
        self, message: ChannelMessage, stream: AsyncIterable[StreamEvent]
    ) -> AsyncGenerator[StreamEvent, None]:
        draft_id = secrets.randbelow(2**31 - 1) + 1
        chat_id = int(message.chat_id)
        text = ""
        last_preview: str | None = None
        last_update = float("-inf")
        draft_enabled = True
        loop = asyncio.get_running_loop()

        async def update_draft(value: str) -> None:
            nonlocal draft_enabled, last_preview, last_update
            if not draft_enabled:
                return
            try:
                rendered = await _render_draft_preview(message, value)
            except Exception as exc:
                draft_enabled = False
                logger.warning("telegram.draft renderer disabled chat_id={} error={}", chat_id, exc)
                return
            if rendered is None or rendered[0] == last_preview:
                return
            plain, html = rendered
            try:
                await self._send_draft(chat_id, draft_id, plain, html)
            except TelegramError as exc:
                draft_enabled = False
                logger.warning("telegram.draft disabled chat_id={} error={}", chat_id, exc)
                return
            last_preview = plain
            last_update = loop.time()

        async for event in stream:
            if event.kind == "text":
                text += str(event.data.get("delta", ""))
                if loop.time() - last_update >= self._settings.stream_update_interval:
                    await update_draft(text)
            elif event.kind == "final":
                await update_draft(text)
            yield event

    async def _on_start(self, update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return
        if self._allow_chats and str(update.message.chat_id) not in self._allow_chats:
            await update.message.reply_text(NO_ACCESS_MESSAGE)
            return
        await update.message.reply_text("Bub is online. Send text to start.")

    async def _on_message(self, update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.effective_user is None:
            return
        chat_id = str(update.message.chat_id)
        if self._allow_chats and chat_id not in self._allow_chats:
            return
        user = update.effective_user
        sender_tokens = {str(user.id)}
        if user.username:
            sender_tokens.add(user.username)
        if self._allow_users and sender_tokens.isdisjoint(self._allow_users):
            await update.message.reply_text("Access denied.")
            return
        await self._on_receive(await self._build_message(update.message))

    async def _build_message(self, message: Message) -> ChannelMessage:
        chat_id = str(message.chat_id)
        session_id = f"{self.name}:{chat_id}"
        content, metadata = await self._parser.parse(message)
        if content.startswith("/bub "):
            content = content[5:]

        # Pass comma commands directly to the input handler
        if content.strip().startswith(","):
            return ChannelMessage(session_id=session_id, content=content.strip(), channel=self.name, chat_id=chat_id)

        media_items = _extract_media_items(metadata)
        reply_meta = await self._parser.get_reply(message)
        if reply_meta:
            metadata["reply_to_message"] = reply_meta
            reply_media = _extract_media_items(reply_meta)
            media_items.extend(reply_media)
        content = json.dumps({"message": content, **metadata}, ensure_ascii=False)
        is_active = MESSAGE_FILTER.filter(message) is not False
        chat_type = getattr(getattr(message, "chat", None), "type", None)
        return ChannelMessage(
            session_id=session_id,
            channel=self.name,
            chat_id=chat_id,
            content=content,
            context=exclude_none({"chat_type": chat_type}),
            media=media_items,
            is_active=is_active,
            lifespan=self.start_typing(chat_id),
            output_channel="null",  # disable outbound for telegram messages
        )

    @contextlib.asynccontextmanager
    async def start_typing(self, chat_id: str) -> AsyncGenerator[None, None]:
        if chat_id in self._typing_tasks:
            yield
            return
        task = asyncio.create_task(self._typing_loop(chat_id))
        self._typing_tasks[chat_id] = task
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            del self._typing_tasks[chat_id]

    async def _typing_loop(self, chat_id: str) -> None:
        while True:
            try:
                await self._app.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(4)  # Telegram typing status lasts for 5 seconds, so we refresh it every 4 seconds
            except Exception as e:
                logger.error(f"Error in typing loop for chat_id={chat_id}: {e}")
                break


class TelegramMessageParser:
    def __init__(self, bot_getter: Callable[[], Bot] | None = None) -> None:
        self._bot_getter = bot_getter

    async def parse(self, message: Message) -> tuple[str, dict[str, Any]]:
        msg_type = _message_type(message)
        content, media = f"[Unsupported message type: {msg_type}]", None
        if msg_type == "text":
            content, media = getattr(message, "text", None) or "", None
        else:
            parser = getattr(self, f"_parse_{msg_type}", None)
            if parser is not None:
                content, media = await parser(message)
        metadata = exclude_none({
            "message_id": message.message_id,
            "type": _message_type(message),
            "username": message.from_user.username if message.from_user else "",
            "full_name": message.from_user.full_name if message.from_user else "",
            "sender_id": str(message.from_user.id) if message.from_user else "",
            "sender_is_bot": message.from_user.is_bot if message.from_user else None,
            "date": message.date.timestamp() if message.date else None,
            "links": self._extract_links(message),
            "media": media,
        })
        return content, metadata

    async def get_reply(self, message: Message) -> dict[str, Any] | None:
        reply_to = message.reply_to_message
        if reply_to is None or reply_to.from_user is None:
            return None
        content, metadata = await self.parse(reply_to)
        return {"message": content, **metadata}

    @staticmethod
    def _extract_links(message: Message) -> list[str] | None:
        entities = getattr(message, "entities", None)
        source_text = getattr(message, "text", None) or ""
        if not entities:
            entities = getattr(message, "caption_entities", None)
            source_text = getattr(message, "caption", None) or ""
        if not entities:
            return None

        links: list[str] = []
        for entity in entities:
            url: str | None = None
            if entity.type == "text_link":
                url = getattr(entity, "url", None)
            elif entity.type == "url":
                offset = getattr(entity, "offset", 0)
                length = getattr(entity, "length", 0)
                candidate = source_text[offset : offset + length].strip()
                url = candidate or None

            if url and url not in links:
                links.append(url)
        return links or None

    async def _parse_photo(self, message: Message) -> tuple[str, dict[str, Any] | None]:
        caption = getattr(message, "caption", None) or ""
        formatted = f"[Photo message] Caption: {caption}" if caption else "[Photo message]"
        photos = getattr(message, "photo", None) or []
        if not photos:
            return formatted, None
        largest = photos[-1]
        mime_type = "image/jpeg"
        media = exclude_none({
            "file_id": largest.file_id,
            "file_size": largest.file_size,
            "width": largest.width,
            "height": largest.height,
            "mime_type": mime_type,
            "data_fetcher": lambda: self._download_media(largest.file_id, largest.file_size),
        })
        return formatted, media

    async def _parse_audio(self, message: Message) -> tuple[str, dict[str, Any] | None]:
        audio = getattr(message, "audio", None)
        if audio is None:
            return "[Audio]", None
        title = audio.title or "Unknown"
        performer = audio.performer or ""
        duration = audio.duration or 0
        metadata = exclude_none({
            "file_id": audio.file_id,
            "mime_type": audio.mime_type,
            "file_size": audio.file_size,
            "duration": audio.duration,
            "title": audio.title,
            "performer": audio.performer,
            "data_fetcher": lambda: self._download_media(audio.file_id, audio.file_size),
        })
        if performer:
            return f"[Audio: {performer} - {title} ({duration}s)]", metadata
        return f"[Audio: {title} ({duration}s)]", metadata

    async def _download_media(self, file_id: str, file_size: int) -> bytes | None:
        if not file_id:
            raise ValueError("file_id must not be empty")
        if self._bot_getter is None:
            raise RuntimeError("Telegram bot is not configured for media downloads.")
        if file_size > 2 * 1024 * 1024:  # limit to 2MB
            return None
        bot = self._bot_getter()
        if bot is None:
            raise RuntimeError("Telegram bot is not available for media downloads.")

        telegram_file = await bot.get_file(file_id)
        if telegram_file is None:
            raise RuntimeError(f"Telegram file lookup returned no result for file_id={file_id}.")
        data = await telegram_file.download_as_bytearray()
        return bytes(data)

    async def _parse_sticker(self, message: Message) -> tuple[str, dict[str, Any] | None]:
        sticker = getattr(message, "sticker", None)
        if sticker is None:
            return "[Sticker]", None
        emoji = sticker.emoji or ""
        set_name = sticker.set_name or ""
        mime_type = "image/webp" if not sticker.is_animated else "video/webm"
        metadata = exclude_none({
            "file_id": sticker.file_id,
            "width": sticker.width,
            "height": sticker.height,
            "mime_type": mime_type,
            "emoji": sticker.emoji,
            "set_name": sticker.set_name,
            "is_animated": sticker.is_animated,
            "data_fetcher": lambda: self._download_media(sticker.file_id, sticker.file_size),
        })
        if emoji:
            return f"[Sticker: {emoji} from {set_name}]", metadata
        return f"[Sticker from {set_name}]", metadata

    async def _parse_video(self, message: Message) -> tuple[str, dict[str, Any] | None]:
        video = getattr(message, "video", None)
        duration = video.duration if video else 0
        caption = getattr(message, "caption", None) or ""
        formatted = f"[Video: {duration}s]"
        formatted = f"{formatted} Caption: {caption}" if caption else formatted
        if video is None:
            return formatted, None
        metadata = exclude_none({
            "file_id": video.file_id,
            "file_size": video.file_size,
            "width": video.width,
            "height": video.height,
            "duration": video.duration,
            "mime_type": video.mime_type,
            "data_fetcher": lambda: self._download_media(video.file_id, video.file_size),
        })
        return formatted, metadata

    async def _parse_voice(self, message: Message) -> tuple[str, dict[str, Any] | None]:
        voice = getattr(message, "voice", None)
        duration = voice.duration if voice else 0
        if voice is None:
            return f"[Voice message: {duration}s]", None
        metadata = exclude_none({
            "file_id": voice.file_id,
            "duration": voice.duration,
            "mime_type": voice.mime_type or "audio/ogg",
            "data_fetcher": lambda: self._download_media(voice.file_id, voice.file_size),
        })
        return f"[Voice message: {duration}s]", metadata

    async def _parse_document(self, message: Message) -> tuple[str, dict[str, Any] | None]:
        document = getattr(message, "document", None)
        if document is None:
            return "[Document]", None
        file_name = document.file_name or "unknown"
        mime_type = document.mime_type or "application/octet-stream"
        caption = getattr(message, "caption", None) or ""
        formatted = f"[Document: {file_name} ({mime_type})]"
        formatted = f"{formatted} Caption: {caption}" if caption else formatted
        metadata = exclude_none({
            "file_id": document.file_id,
            "file_name": document.file_name,
            "file_size": document.file_size,
            "mime_type": mime_type,
            "data_fetcher": lambda: self._download_media(document.file_id, document.file_size),
        })
        return formatted, metadata

    async def _parse_video_note(self, message: Message) -> tuple[str, dict[str, Any] | None]:
        video_note = getattr(message, "video_note", None)
        duration = video_note.duration if video_note else 0
        if video_note is None:
            return f"[Video note: {duration}s]", None
        metadata = exclude_none({
            "file_id": video_note.file_id,
            "duration": video_note.duration,
            "mime_type": video_note.mime_type or "video/mp4",
            "data_fetcher": lambda: self._download_media(video_note.file_id, video_note.file_size),
        })
        return f"[Video note: {duration}s]", metadata
