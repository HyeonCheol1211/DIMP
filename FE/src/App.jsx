import React, { useState, useRef, useEffect } from "react";
import './App.css';

export default function App() {
  const [input, setInput] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const [imagePreview, setImagePreview] = useState(null);
  const chatEndRef = useRef(null);

  const fileToBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });

  const sendMessage = async () => {
    if (!input.trim() && !imagePreview) return;

    if (input.trim())
      setChatLog((prev) => [...prev, { sender: "user", text: input }]);
    if (imagePreview)
      setChatLog((prev) => [...prev, { sender: "user", image: imagePreview }]);

    try {
      const body = {
        ...(input.trim() && { message: input }),
        ...(imagePreview && { image_url: imagePreview }),
      };

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (data.reply) {
        setChatLog((prev) => [...prev, { sender: "bot", text: data.reply }]);
      }
      if (data.reply_image_url) {
        setChatLog((prev) => [
          ...prev,
          { sender: "bot", image: data.reply_image_url },
        ]);
      }
    } catch (error) {
      setChatLog((prev) => [
        ...prev,
        { sender: "bot", text: "서버와 통신 중 오류가 발생했습니다." },
      ]);
    }

    setInput("");
    setImagePreview(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleImageChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const base64 = await fileToBase64(file);
    setImagePreview(base64);
    e.target.value = null;
  };

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatLog]);

  const canSend = input.trim() || imagePreview;

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-logo">DIMP</div>
      </header>

      <main className="chatbox" role="main" tabIndex="-1">
        <div className="chatlog" aria-live="polite" aria-relevant="additions">
          {chatLog.length === 0 && (
            <div className="chat-empty">피부질환을 케어해드려요!</div>
          )}
          {chatLog.map((msg, i) => (
            <div
              key={i}
              className={`chat-message ${msg.sender === "user" ? "user" : "bot"}`}
              aria-label={`${msg.sender === "user" ? "사용자" : "봇"} 메시지`}
            >
              {msg.text && <p className="message-text">{msg.text}</p>}
              {msg.image && (
                <img
                  src={msg.image}
                  alt="전송된 이미지"
                  className="message-image"
                  loading="lazy"
                />
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        {imagePreview && (
          <div className="image-preview-container">
            <img
              src={imagePreview}
              alt="첨부된 이미지"
              className="message-image"
            />
            <button
              onClick={() => setImagePreview(null)}
              className="remove-image-button"
              title="이미지 제거"
            >
              X
            </button>
          </div>
        )}

        <form
          className="input-area"
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
          aria-label="메시지 입력창"
        >
          <label htmlFor="image-upload" className="image-upload-label" title="사진 첨부">
            <img src="/images/picture2.png" alt="사진 첨부 이미지" className="image-icon" />
          </label>
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            style={{ display: "none" }}
            aria-hidden="true"
          />

          <textarea
            rows={1}
            placeholder="문의할 내용을 입력해주세요."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            className="input-textarea"
            aria-multiline="true"
          />

          <button
            type="submit"
            className="send-button"
            title="전송"
            disabled={!canSend}
          >
            <img
              src="/images/send.png"
              alt="전송"
              className="send-icon"
            />
          </button>
        </form>
      </main>
    </div>
  );
}
