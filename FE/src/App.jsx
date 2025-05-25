// frontend/src/App.js
import React, { useState } from "react";

function App() {
  const [input, setInput] = useState("");
  const [chatLog, setChatLog] = useState([]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // 사용자 메시지 추가
    setChatLog((prev) => [...prev, { sender: "user", text: input }]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
      const data = await response.json();
      // 챗봇 응답 추가
      setChatLog((prev) => [...prev, { sender: "bot", text: data.reply }]);
    } catch (error) {
      setChatLog((prev) => [
        ...prev,
        { sender: "bot", text: "서버와 통신 중 오류가 발생했습니다." },
      ]);
    }

    setInput("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div style={{ maxWidth: 600, margin: "20px auto", fontFamily: "Arial" }}>
      <h2>FastAPI + React 챗봇</h2>
      <div
        style={{
          border: "1px solid #ccc",
          height: 400,
          padding: 10,
          overflowY: "auto",
          marginBottom: 10,
        }}
      >
        {chatLog.map((msg, i) => (
          <div
            key={i}
            style={{
              textAlign: msg.sender === "user" ? "right" : "left",
              margin: "8px 0",
            }}
          >
            <span
              style={{
                display: "inline-block",
                padding: "8px 12px",
                borderRadius: 15,
                backgroundColor: msg.sender === "user" ? "#007bff" : "#e4e6eb",
                color: msg.sender === "user" ? "white" : "black",
                maxWidth: "80%",
              }}
            >
              {msg.text}
            </span>
          </div>
        ))}
      </div>
      <input
        type="text"
        placeholder="메시지를 입력하세요"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={handleKeyPress}
        style={{ width: "100%", padding: 10, fontSize: 16 }}
      />
      <button onClick={sendMessage} style={{ marginTop: 10, width: "100%" }}>
        전송
      </button>
    </div>
  );
}

export default App;
