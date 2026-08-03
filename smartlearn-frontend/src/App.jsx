import { useState } from "react"
import { uploadPDF } from "./api"
import PdfPreview from "./PdfPreview"
import ChatPanel from "./ChatPanel"

function App() {
  const [file, setFile] = useState(null)
  const [upload, setUpload] = useState(null)
  const [activePage, setActivePage] = useState(1)
  const [uploadKey, setUploadKey] = useState(0)
  const [status, setStatus] = useState("idle")
  const [error, setError] = useState("")

  async function handleUpload(event) {
    event.preventDefault()
    if (!file) return
    try {
      setStatus("uploading")
      setError("")
      const result = await uploadPDF(file)
      setUpload(result)
      setActivePage(1)
      setUploadKey((key) => key + 1)
    } catch (err) {
      setError(err.message || "Upload failed.")
    } finally {
      setStatus("idle")
    }
  }

  function handleJumpToPage(page) {
    setActivePage(page)
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

      {/* ---- 工作区: 左预览 + 右对话 ---- */}
      <div className="workspace">
        <PdfPreview
          upload={upload}
          activePage={activePage}
          previewKey={uploadKey}
        />
        <ChatPanel
          key={uploadKey}
          enabled={!!upload}
          onBusy={(busy) => setStatus(busy ? "asking" : "idle")}
          disabled={status === "uploading"}
          onJumpToPage={handleJumpToPage}
        />
      </div>
    </main>
  )
}

export default App
