export default function SentimentBar({ distribution = { positive: 0, neutral: 0, negative: 0 } }) {
  const rows = [
    { key: 'positive', label: 'Positive', color: 'bg-green-500', value: distribution.positive || 0 },
    { key: 'neutral', label: 'Neutral', color: 'bg-gray-400', value: distribution.neutral || 0 },
    { key: 'negative', label: 'Negative', color: 'bg-red-500', value: distribution.negative || 0 },
  ]

  return (
    <div className="space-y-3">
      {rows.map((row) => (
        <div key={row.key}>
          <div className="mb-1 flex items-center justify-between text-xs text-gray-600">
            <span>{row.label}</span>
            <span>{row.value}%</span>
          </div>
          <div className="h-3 w-full rounded-full bg-gray-100">
            <div
              className={`h-3 rounded-full ${row.color} transition-all duration-500`}
              style={{ width: `${Math.max(0, Math.min(100, row.value))}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
