import { useNavigate } from 'react-router-dom'
import SearchBar from '../components/SearchBar'

const quickConditions = ['Depression', 'Anxiety', 'Birth Control', 'Diabetes', 'ADHD', 'Pain Relief']

export default function Home() {
  const navigate = useNavigate()

  const handleSearch = (drug, condition) => {
    if (!drug || !condition) {
      return
    }
    navigate(`/search?drug=${encodeURIComponent(drug)}&condition=${encodeURIComponent(condition)}`)
  }

  return (
    <section className="mx-auto flex max-w-4xl flex-col items-center px-4 py-16 text-center">
      <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">Understand any drug in seconds</h1>
      <p className="mt-4 text-base text-gray-600">Search from thousands of real patient reviews</p>
      <div className="mt-8 w-full">
        <SearchBar onSearch={handleSearch} />
      </div>
      <div className="mt-8 flex flex-wrap justify-center gap-2">
        {quickConditions.map((condition) => (
          <button
            key={condition}
            type="button"
            onClick={() => navigate(`/search?condition=${encodeURIComponent(condition)}`)}
            className="rounded-full border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700 hover:border-indigo-400 hover:text-indigo-600"
          >
            {condition}
          </button>
        ))}
      </div>
    </section>
  )
}
