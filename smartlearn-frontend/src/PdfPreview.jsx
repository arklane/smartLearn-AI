import { getDocumentFileURL } from "./api"

function PdfPreview({ upload, activePage, previewKey }) {
  if (!upload) {
    return (
      <div className="preview-panel">
        <div className="preview-placeholder">
          Upload a PDF to see the preview here.
        </div>
      </div>
    )
  }

  const page = activePage || 1
  const url = getDocumentFileURL(page)

  return (
    <div className="preview-panel">
      <div className="preview-header">
        <span className="preview-filename">{upload.filename}</span>
        <span className="preview-page-label">Page {page}</span>
      </div>
      <iframe
        key={`${previewKey}-${page}`}
        title="PDF preview"
        src={url}
        className="preview-frame"
      />
    </div>
  )
}

export default PdfPreview
