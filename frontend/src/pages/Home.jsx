import { useNavigate } from 'react-router-dom'
import SearchBar from '../components/SearchBar'

const conditions = ['Depression', 'Anxiety', 'Birth Control', 'Diabetes', 'ADHD', 'Pain Relief']

export default function Home() {
  const navigate = useNavigate()

  const onSearch = (drug, condition) => {
    navigate(`/search?drug=${encodeURIComponent(drug)}&condition=${encodeURIComponent(condition)}`)
  }

  return (
    <section className="mx-auto flex max-w-6xl flex-col items-center px-4 py-16 text-center sm:px-6">
      <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">Understand any drug in seconds</h1>
      <p className="mt-4 text-base text-gray-600 sm:text-lg">Search from thousands of real patient reviews</p>
      <div className="mt-8 w-full">
        <SearchBar onSearch={onSearch} />
      </div>

      <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
        {conditions.map((condition) => (
          <button
            key={condition}
            type="button"
            className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-sm font-medium text-indigo-700 hover:bg-indigo-100"
            onClick={() => navigate(`/search?drug=&condition=${encodeURIComponent(condition)}`)}
          >
            {condition}
          </button>
        ))}
      </div>
    </section>
  )
}
