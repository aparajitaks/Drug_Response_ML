function badgeClass(label) {
  if (label === 'Responder') return 'bg-emerald-100 text-emerald-700'
  if (label === 'Non-Responder') return 'bg-red-100 text-red-700'
  return 'bg-gray-200 text-gray-700'
}

export default function PredictionCard({ prediction_label, prediction_class, confidence }) {
  const pct = Math.round((confidence || 0) * 100)

  return (
    <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-gray-900">Prediction Result</h2>
      <div className="mt-3 flex items-center gap-3">
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${badgeClass(prediction_label)}`}>
          {prediction_label}
        </span>
        <span className="text-sm text-gray-600">class: {prediction_class}</span>
      </div>

      <div className="mt-5">
        <div className="mb-1 flex justify-between text-sm text-gray-700">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div className="h-2.5 rounded-full bg-gray-200">
          <div className="h-full rounded-full bg-indigo-600 transition-all duration-500" style={{ width: `${pct}%` }} />
        </div>
      </div>
    </section>
  )
}
