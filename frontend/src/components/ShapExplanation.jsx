export default function ShapExplanation({ shap_explanation = [] }) {
  return (
    <section className="rounded-xl bg-white p-5 shadow-sm">
      <h3 className="mb-3 text-lg font-semibold text-gray-900">Why this prediction?</h3>
      <div className="space-y-2">
        {shap_explanation.slice(0, 3).map((item, index) => {
          const positive = item.direction === 'positive'
          return (
            <div key={`${item.feature}-${index}`} className="flex items-center justify-between rounded-lg bg-gray-50 p-3">
              <div className="flex items-center gap-2">
                <span className={`text-sm font-bold ${positive ? 'text-green-600' : 'text-red-600'}`}>
                  {positive ? '↑' : '↓'}
                </span>
                <span className="text-sm text-gray-700">{item.feature}</span>
              </div>
              <span className="text-xs font-medium uppercase tracking-wide text-gray-500">{item.impact}</span>
            </div>
          )
        })}
        {shap_explanation.length === 0 && <p className="text-sm text-gray-500">No explanation available.</p>}
      </div>
    </section>
  )
}
