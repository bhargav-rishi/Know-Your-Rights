import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./index.css";
import { FaUser, FaRobot, FaTrash, FaMoon, FaSun } from "react-icons/fa";
import { motion } from "framer-motion";
import { Tooltip } from "react-tooltip";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [isDelayed, setIsDelayed] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [fileInputKey, setFileInputKey] = useState(Date.now()); // for clearing file input
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const api = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";




  const chatRef = useRef(null);

  useEffect(() => {
    chatRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

const handleSend = async () => {
  if (!input.trim()) return;

  const userMessage = {
    role: "user",
    text: input,
    timestamp: new Date().toLocaleTimeString(),
  };

  setMessages((prev) => [...prev, userMessage]);
  setInput("");
  setIsTyping(true);
  setIsDelayed(false);

  // Set a delay indicator after 10 seconds
  const delayTimer = setTimeout(() => {
    setIsDelayed(true);
  }, 10000); // 10 seconds

  try {
    const res = await axios.post(`${api}/chat`, {
      question: input,
    });

    clearTimeout(delayTimer);
    setIsDelayed(false);

    const botMessage = {
      role: "bot",
      text: res.data?.answer || "No answer returned.",
      sources: res.data?.sources || [],
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, botMessage]);
  } catch (err) {
    clearTimeout(delayTimer);
    setIsDelayed(false);

    setMessages((prev) => [
      ...prev,
      {
        role: "bot",
        text: err.response?.data?.answer || "Sorry, something went wrong. Please try again later.",
        timestamp: new Date().toLocaleTimeString(),
      },
    ]);
  } finally {
    setIsTyping(false);
  }
};


  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSend();
  };

  const clearChat = () => {
    setMessages([]);
  };

  const toggleTheme = () => {
    setDarkMode((prev) => !prev);
  };

useEffect(() => {
  document.documentElement.classList.toggle("dark", darkMode);
}, [darkMode]);

const handleFileUpload = async (e) => {
  const files = Array.from(e.target.files);
  const maxFiles = 5;
  const maxSize = 10 * 1024 * 1024; // 10 MB

  // Clear previous status
  setUploadMessage("");
  setUploadSuccess(false);

  // 1. Validate file count
  if (files.length > maxFiles) {
    setUploadMessage(`⚠️ You can only upload up to ${maxFiles} PDFs at a time.`);
    return;
  }

  // 2. Validate file type and size
  for (let f of files) {
    if (!f.name.toLowerCase().endsWith(".pdf")) {
      setUploadMessage("⚠️ Only PDF files are allowed.");
      return;
    }
    if (f.size > maxSize) {
      setUploadMessage(`⚠️ File "${f.name}" exceeds the 10MB limit.`);
      return;
    }
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  try {
    setUploading(true);
    setUploadMessage("📤 Uploading...");
    setUploadedFiles(files);

    await axios.post(`${api}/upload`, formData);

    setUploadSuccess(true);
    setUploadMessage("✅ Upload successful!");
    setFileInputKey(Date.now()); // reset file input
  } catch (error) {
    console.error("❌ Upload error:", error);
    setUploadMessage("❌ Upload failed. Try again.");
    setUploadSuccess(false);
  } finally {
    setUploading(false);
  }
};


return (
  <div className="min-h-screen transition-all duration-500 ease-in-out bg-gradient-to-br from-blue-500 to-blue-200 dark:from-gray-600 dark:to-gray-900 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-900 dark:text-white transition-all duration-500 ease-in-out rounded-2xl shadow-2xl p-6 w-full max-w-3xl flex flex-col gap-4">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold text-center flex-grow">
            Know Your Rights - Legal Chatbot
          </h1>
          <button
            onClick={toggleTheme}
            className="ml-4 text-xl text-yellow-600 dark:text-yellow-300 hover:scale-110 transition"
            title="Toggle Theme"
          >
            {darkMode ? <FaSun /> : <FaMoon />}
          </button>
        </div>

        <div className="chat-box flex flex-col gap-3 overflow-y-auto max-h-[500px] pr-2">
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`message flex items-start gap-2 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "bot" && <div className="avatar text-xl pt-1"><FaRobot /></div>}
              <div className={`bubble ${msg.role === "user" ? "bg-blue-100 dark:bg-blue-900 text-gray-900 dark:text-white text-right self-end": "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white"} p-3 rounded-xl shadow-md max-w-[80%]`}>
                <div className="text text-gray-900 dark:text-white whitespace-pre-line">
                  {msg.text}
                </div>

                {msg.sources?.length > 0 && (
                  <div className="sources mt-2 text-sm text-gray-600 dark:text-gray-300">
                    <strong>Sources:</strong>
                    <ul className="list-disc list-inside">
                      {msg.sources.map((s, i) => (
                        <li key={i} className="ml-2 text-sm">
                          📄 <strong>Source</strong>: {s.source || "Unknown"}<br />
                          🔍 <strong>Chunk</strong>: <span className="bg-yellow-100 dark:bg-yellow-600 px-1 rounded">
                            {s.chunk?.slice(0, 300)}...
                          </span>
                          {s.score !== undefined && (
                            <>
                              <br />
                              📊 <span data-tip="Semantic similarity score (0 to 1)"><strong>Score</strong>: {s.score.toFixed(3)}</span>
                            </>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="timestamp text-xs text-gray-500 mt-1">
                  {msg.timestamp}
                </div>
              </div>
              {msg.role === "user" && <div className="avatar text-xl pt-1"><FaUser /></div>}
            </motion.div>
          ))}
          {isTyping && (
            <div className="bubble italic text-gray-500 dark:text-gray-300 animate-pulse">
              {isDelayed ? "Still working on your question..." : "Bot is typing..."}
            </div>
          )}
          <div ref={chatRef} />
        </div>
          
{/* --- Clean File Upload UI --- */}
<div className="flex flex-col items-center gap-2">
  <label
    htmlFor="file-upload"
    className={`cursor-pointer text-sm px-4 py-2 rounded-lg border border-dashed border-gray-400 text-center w-full hover:bg-gray-50 dark:hover:bg-gray-800 transition
    ${uploading ? "opacity-60 cursor-not-allowed" : ""}`}
  >
    📎 Click to upload PDF files
    <input
      key={fileInputKey}
      id="file-upload"
      type="file"
      accept="application/pdf"
      multiple
      onChange={handleFileUpload}
      className="hidden"
    />
  </label>

  {uploadMessage && (
  <p className={`text-sm mt-1 ${uploadSuccess ? "text-green-600" : "text-red-500"}`}>
    {uploadMessage}
  </p>
  )}

  {uploadSuccess && uploadedFiles.length > 0 && (
    <div className="text-sm text-gray-700 dark:text-gray-300 mt-1">
      Uploaded: {uploadedFiles.map((f) => f.name).join(", ")}
    </div>
  )}
</div>


        <div className="input-box flex gap-2 ">
          
          <input
            disabled={isTyping || isUploading}
            type="text"
            placeholder="Ask a legal question..."
            className="flex-grow border border-blue-300 rounded-lg px-4 py-2 text-base outline-none focus:ring-2 focus:ring-blue-400 dark:bg-gray-800 dark:text-white"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
            onClick={handleSend}
            disabled={isTyping || isUploading}
          >
            Ask
          </button>
          
          <button
            className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition"
            onClick={clearChat}
            title="Clear Chat"
          >
            <FaTrash />
          </button>
        </div>
        <Tooltip effect="solid" />
      </div>
    </div>
  );
}

export default App;





// function App() {
//   const [messages, setMessages] = useState([]);
//   const [question, setQuestion] = useState("");
//   const [loading, setLoading] = useState(false);

//   const askBot = async () => {
//     if (!question.trim()) return;
//     const newMessage = { type: "user", text: question, timestamp: new Date() };
//     setMessages((prev) => [...prev, newMessage]);
//     setLoading(true);
//     setQuestion("");

//     try {
//       const response = await fetch("http://127.0.0.1:8000/chat", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ question }),
//       });

//       const data = await response.json();
//       const botMessage = {
//         type: "bot",
//         text: data.answer,
//         sources: data.sources || [],
//         timestamp: new Date(),
//       };
//       setMessages((prev) => [...prev, botMessage]);
//     } catch (err) {
//       setMessages((prev) => [
//         ...prev,
//         { type: "bot", text: "Something went wrong. Please try again.", timestamp: new Date() },
//       ]);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="container">
//       <h1>Know Your Rights - Legal Chatbot</h1>
//       <ChatBox messages={messages} loading={loading} />
//       <div className="input-box">
//         <input
//           value={question}
//           onChange={(e) => setQuestion(e.target.value)}
//           onKeyDown={(e) => e.key === "Enter" && askBot()}
//           placeholder="Ask a legal question..."
//         />
//         <button onClick={askBot} disabled={loading}>
//           {loading ? "..." : "Ask"}
//         </button>
//       </div>
//     </div>
//   );
// }


// export default App;