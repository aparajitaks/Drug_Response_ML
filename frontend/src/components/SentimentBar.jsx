function Row({ label, value, colorClass }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="capitalize text-gray-700">{label}</span>
        <span className="font-medium text-gray-600">{value}%</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-gray-200">
        <div
          className={`h-full rounded-full transition-all duration-500 ${colorClass}`}
          style={{ width: `${Math.max(0, Math.min(100, value || 0))}%` }}
        />
      </div>
    </div>
  )
}

export default function SentimentBar({ distribution = { positive: 0, neutral: 0, negative: 0 } }) {
  return (
    <div className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
      <Row label="positive" value={distribution.positive || 0} colorClass="bg-emerald-500" />
      <Row label="neutral" value={distribution.neutral || 0} colorClass="bg-gray-500" />
      <Row label="negative" value={distribution.negative || 0} colorClass="bg-red-500" />
    </div>
  )
}
