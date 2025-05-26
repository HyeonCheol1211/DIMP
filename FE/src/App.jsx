import React, { useState, useRef, useEffect } from "react";
import './App.css';

export default function App() {
  const [input, setInput] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const chatEndRef = useRef(null);

  const fileToBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });

  const sendMessage = async (imageDataUrl = null) => {
    if (!input.trim() && !imageDataUrl) return;

    if (input.trim())
      setChatLog((prev) => [...prev, { sender: "user", text: input }]);
    if (imageDataUrl)
      setChatLog((prev) => [...prev, { sender: "user", image: imageDataUrl }]);

    try {
      const body = imageDataUrl
        ? { image_url: imageDataUrl }
        : { message: input };

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
    sendMessage(base64);
    e.target.value = null;
  };

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatLog]);

  return (
    <div className="app-container">
      {/* 상단 헤더 */}
      <header className="app-header">
        <div className="app-logo">DIMP</div>
      </header>

      {/* 채팅 내용 */}
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

        {/* 입력 영역 */}
        <form
          className="input-area"
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
          aria-label="메시지 입력창"
        >
          <label htmlFor="image-upload" className="image-upload-label" title="사진 첨부">
            📷
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
            placeholder="메시지를 입력하세요"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            className="input-textarea"
            aria-multiline="true"
          />
          <button type="submit" className="send-button" title="전송">
            
          </button>
        </form>
      </main>
    </div>
  );
}
