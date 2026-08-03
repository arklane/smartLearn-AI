import { useState } from "react"
import { askQuestion } from "./api"

function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [message, setMessage] = useState("")
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  async function handleAsk(event) {
    event.preventDefault()
    const text = message.trim()
    if (!text || !enabled || loading) return

    setMessage("")
    setError("")
    setLoading(true)
    if (onBusy) onBusy(true)

    setMessages((prev) => [...prev, { role: "user", content: text }])

    try {
      const result = await askQuestion(text)
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          citations: result.citations || [],
          sources: result.sources || [],
        },
      ])
    } catch (err) {
      setError(err.message || "Chat failed.")
    } finally {
      setLoading(false)
      if (onBusy) onBusy(false)
    }
  }

  return (
    <section className="chat-panel">
      <div className="message-list">
        {messages.length === 0 && !loading && (
          <p className="message-placeholder">
            Ask a question about the uploaded PDF.
          </p>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`message message-${msg.role}`}>
            <p className="message-content">{msg.content}</p>
            {msg.role === "assistant" && msg.citations.length > 0 && (
              <div className="citations">
                {msg.citations.map((page) => (
                  <button
                    key={page}
                    type="button"
                    className="chip citation-chip"
                    onClick={() => onJumpToPage && onJumpToPage(page)}
                  >
                    Page {page}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <p className="status-text">Asking…</p>}
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      <form className="chat-form" onSubmit={handleAsk}>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={enabled ? "Ask a question…" : "Upload a PDF first"}
          disabled={!enabled || loading}
        />
        <button
          type="submit"
          disabled={!enabled || !message.trim() || loading || disabled}
        >
          {loading ? "Asking…" : "Ask"}
        </button>
      </form>
    </section>
  )
}

export default ChatPanel
