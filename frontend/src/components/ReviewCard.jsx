import { useState } from 'react'

export default function ReviewCard({ review = '', usefulCount = 0, rating = null }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <article className="rounded-xl bg-white p-4 shadow-sm">
      <p className={`${expanded ? '' : 'line-clamp-3'} text-sm leading-6 text-gray-700`}>{review}</p>
      <button
        type="button"
        className="mt-2 text-xs font-medium text-indigo-600"
        onClick={() => setExpanded((prev) => !prev)}
      >
        {expanded ? 'Show less' : 'Read more'}
      </button>
      <div className="mt-3 flex items-center justify-between">
        <span className="rounded-full bg-blue-100 px-2.5 py-1 text-xs font-medium text-blue-700">
          usefulCount: {usefulCount}
        </span>
        <span className="text-xs text-gray-600">rating: {rating ?? '-'} / 10</span>
      </div>
    </article>
  )
}
