function Arrow({ positive }) {
  return (
    <span className={positive ? 'text-emerald-600' : 'text-red-600'}>{positive ? '↑' : '↓'}</span>
  )
}

export default function ShapExplanation({ shap_explanation = [] }) {
  return (
    <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-gray-900">Why this prediction?</h2>
      <div className="mt-4 space-y-3">
        {shap_explanation.slice(0, 3).map((item, idx) => (
          <div key={`${item.feature}-${idx}`} className="flex items-center justify-between rounded-md border border-gray-200 p-3">
            <div className="flex items-center gap-2">
              <Arrow positive={item.direction === 'positive'} />
              <span className="text-sm font-medium text-gray-800">{item.feature}</span>
            </div>
            <span className="text-xs uppercase tracking-wide text-gray-500">{item.impact}</span>
          </div>
        ))}
        {shap_explanation.length === 0 ? <p className="text-sm text-gray-500">No explanation data available.</p> : null}
      </div>
    </section>
  )
}
