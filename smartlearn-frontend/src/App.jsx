import { useState } from "react"
import { uploadPDF, askQuestion } from "./api"

function App() {
  const [file, setFile] = useState(null)
  const [upload, setUpload] = useState(null)
  const [message, setMessage] = useState("")
  const [answer, setAnswer] = useState(null)
  const [status, setStatus] = useState("idle")
  const [error, setError] = useState("")

  async function handleUpload(event) {
    event.preventDefault()
    if (!file) return
    try {
      setStatus("uploading")
      setError("")
      setAnswer(null)
      const result = await uploadPDF(file)
      setUpload(result)
    } catch (err) {
      setError(err.message || "Upload failed.")
    } finally {
      setStatus("idle")
    }
  }

  async function handleAsk(event) {
    event.preventDefault()
    if (!message.trim()) return
    try {
      setStatus("asking")
      setError("")
      const result = await askQuestion(message.trim())
      setAnswer(result)
    } catch (err) {
      setError(err.message || "Chat failed.")
    } finally {
      setStatus("idle")
    }
  }

  return (
    <main>
      <h1>SmartLearn Lite</h1>

      {/* ---- 上传区域 ---- */}
      <div className="card">
        <form onSubmit={handleUpload}>
          <label htmlFor="pdf-file">Choose PDF</label>
          <input
            id="pdf-file"
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0] || null)}
          />
          <button type="submit" disabled={!file || status !== "idle"}>
            {status === "uploading" ? "Uploading…" : "Upload"}
          </button>
          {status === "uploading" && <span className="status-text">Processing your PDF…</span>}
        </form>
      </div>

      {/* ---- 上传成功信息 ---- */}
      {upload && (
        <p className="upload-info">
          Uploaded <strong>{upload.filename}</strong>: {upload.pages} pages, {upload.characters} characters
        </p>
      )}

      {/* ---- 错误显示 ---- */}
      {error && <p className="error" role="alert">{error}</p>}

      {/* ---- 提问区域 ---- */}
      <div className="card">
        <form onSubmit={handleAsk}>
          <label htmlFor="message">Message</label>
          <textarea
            id="message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder={upload ? "Ask a question about the PDF…" : "Upload a PDF first"}
          />
          <button
            type="submit"
            disabled={!upload || !message.trim() || status !== "idle"}
          >
            {status === "asking" ? "Asking…" : "Ask"}
          </button>
          {status === "asking" && <span className="status-text">Waiting for answer…</span>}
        </form>
      </div>

      {/* ---- 答案显示 ---- */}
      {answer && (
        <section className="answer-section">
          <h2>Answer</h2>
          <p className="answer-text">{answer.answer}</p>
          {answer.citations && answer.citations.length > 0 && (
            <div className="citations">
              {answer.citations.map((page) => (
                <span key={page} className="chip">
                  Page {page}
                </span>
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  )
}

export default App