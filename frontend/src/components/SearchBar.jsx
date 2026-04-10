import { useState } from 'react'

export default function SearchBar({ onSearch, initialDrug = '', initialCondition = '' }) {
  const [drug, setDrug] = useState(initialDrug)
  const [condition, setCondition] = useState(initialCondition)

  const submit = () => {
    onSearch?.(drug.trim(), condition.trim())
  }

  const onKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="w-full rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="grid gap-3 sm:grid-cols-2">
        <input
          value={drug}
          onChange={(e) => setDrug(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Drug Name"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-500"
        />
        <input
          value={condition}
          onChange={(e) => setCondition(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Condition"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-500"
        />
      </div>
      <button
        type="button"
        onClick={submit}
        className="mt-3 w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
      >
        Search
      </button>
    </div>
  )
}
