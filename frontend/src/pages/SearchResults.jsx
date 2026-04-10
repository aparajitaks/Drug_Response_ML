import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { searchDrug } from '../api/drugApi'
import Spinner from '../components/Spinner'
import SentimentBar from '../components/SentimentBar'
import ReviewCard from '../components/ReviewCard'

export default function SearchResults() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const drug = params.get('drug') || ''
  const condition = params.get('condition') || ''

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [compareWith, setCompareWith] = useState('')
  const [showCompareInput, setShowCompareInput] = useState(false)

  useEffect(() => {
    if (!condition) return
    const load = async () => {
      setLoading(true)
      setError('')
      try {
        const response = await searchDrug(drug, condition)
        setData(response.data)
      } catch (err) {
        setError(err?.response?.data?.detail || err.message || 'Search request failed.')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [drug, condition])

  const goCompare = () => {
    if (!compareWith.trim()) return
    navigate(
      `/compare?drug1=${encodeURIComponent(drug)}&drug2=${encodeURIComponent(
        compareWith.trim(),
      )}&condition=${encodeURIComponent(condition)}`,
    )
  }

  if (loading) return <Spinner label="Searching reviews..." />

  return (
    <section className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      {error ? <p className="rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}
      {data ? (
        <>
          <h1 className="text-2xl font-bold text-gray-900 sm:text-3xl">
            {drug || 'All drugs'} for {condition}
          </h1>
          <p className="mt-1 text-sm text-gray-500">Total reviews: {data.total_reviews}</p>

          <div className="mt-5">
            <SentimentBar distribution={data.sentiment_distribution} />
          </div>

          <div className="mt-6 space-y-3">
            {data.top_reviews.map((item, idx) => (
              <ReviewCard key={idx} review={item.review} usefulCount={item.usefulCount} rating={item.rating} />
            ))}
          </div>

          <div className="mt-8">
            <button
              type="button"
              onClick={() => setShowCompareInput((v) => !v)}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
            >
              Compare with another drug
            </button>
            {showCompareInput ? (
              <div className="mt-3 flex max-w-md gap-2">
                <input
                  type="text"
                  value={compareWith}
                  onChange={(e) => setCompareWith(e.target.value)}
                  placeholder="Enter second drug"
                  className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm outline-none ring-indigo-500 focus:ring"
                />
                <button
                  type="button"
                  onClick={goCompare}
                  className="rounded-md border border-indigo-200 bg-indigo-50 px-3 py-2 text-sm font-medium text-indigo-700 hover:bg-indigo-100"
                >
                  Go
                </button>
              </div>
            ) : null}
          </div>
        </>
      ) : null}
    </section>
  )
}
