const styleMap = {
  Responder: 'bg-green-100 text-green-700',
  'Non-Responder': 'bg-red-100 text-red-700',
  'Neutral/Mixed': 'bg-gray-200 text-gray-700',
}

export default function PredictionCard({ prediction_label, prediction_class, confidence }) {
  const normalizedConfidence = Math.max(0, Math.min(100, Math.round((confidence || 0) * 100)))
  const toneClass = styleMap[prediction_label] || 'bg-gray-200 text-gray-700'

  return (
    <section className="rounded-xl bg-white p-5 shadow-sm">
      <h3 className="mb-3 text-lg font-semibold text-gray-900">Prediction Result</h3>
      <div className="mb-3 flex items-center justify-between">
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${toneClass}`}>
          {prediction_label || 'Unknown'}
        </span>
        <span className="text-sm text-gray-600">class: {prediction_class}</span>
      </div>
      <div>
        <div className="mb-1 flex items-center justify-between text-sm text-gray-600">
          <span>Confidence</span>
          <span>{normalizedConfidence}%</span>
        </div>
        <div className="h-3 w-full rounded-full bg-gray-100">
          <div
            className="h-3 rounded-full bg-indigo-600 transition-all duration-500"
            style={{ width: `${normalizedConfidence}%` }}
          />
        </div>
      </div>
    </section>
  )
}
