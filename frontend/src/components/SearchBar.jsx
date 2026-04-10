import { useState } from 'react'

export default function SearchBar({ onSearch, defaultDrug = '', defaultCondition = '' }) {
  const [drug, setDrug] = useState(defaultDrug)
  const [condition, setCondition] = useState(defaultCondition)

  const submit = (event) => {
    event.preventDefault()
    onSearch(drug.trim(), condition.trim())
  }

  return (
    <form onSubmit={submit} className="w-full max-w-3xl rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="grid gap-3 sm:grid-cols-3">
        <input
          type="text"
          value={drug}
          onChange={(e) => setDrug(e.target.value)}
          placeholder="Drug Name"
          className="rounded-md border border-gray-300 px-3 py-2 text-sm outline-none ring-indigo-500 focus:ring"
        />
        <input
          type="text"
          value={condition}
          onChange={(e) => setCondition(e.target.value)}
          placeholder="Condition"
          className="rounded-md border border-gray-300 px-3 py-2 text-sm outline-none ring-indigo-500 focus:ring"
        />
        <button
          type="submit"
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-700"
        >
          Search
        </button>
      </div>
    </form>
  )
}
