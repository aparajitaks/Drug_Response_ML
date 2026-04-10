import { useState } from 'react'

export default function ReviewCard({ review, usefulCount, rating }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <article className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <p
        className="text-sm text-gray-700"
        style={
          expanded
            ? undefined
            : {
                display: '-webkit-box',
                WebkitLineClamp: 3,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
              }
        }
      >
        {review}
      </p>
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="mt-2 text-xs font-medium text-indigo-600 hover:text-indigo-700"
      >
        {expanded ? 'Show less' : 'Read more'}
      </button>
      <div className="mt-3 flex items-center justify-between">
        <span className="rounded-full bg-blue-100 px-2.5 py-1 text-xs font-medium text-blue-700">
          usefulCount: {usefulCount ?? 0}
        </span>
        <span className="text-xs text-gray-600">rating: {rating ?? 0}/10</span>
      </div>
    </article>
  )
}
