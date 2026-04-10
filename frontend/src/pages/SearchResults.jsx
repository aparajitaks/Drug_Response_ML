import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { searchDrug } from '../api/drugApi'
import ReviewCard from '../components/ReviewCard'
import SentimentBar from '../components/SentimentBar'
import Spinner from '../components/Spinner'

export default function SearchResults() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const drug = (params.get('drug') || '').trim()
  const condition = (params.get('condition') || '').trim()

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [data, setData] = useState(null)
  const [compareDrug, setCompareDrug] = useState('')
  const [showCompareInput, setShowCompareInput] = useState(false)

  useEffect(() => {
    let active = true

    const run = async () => {
      setLoading(true)
      setError('')
      if (!drug || !condition) {
        setError('Please provide both drug and condition in the search query.')
        setLoading(false)
        return
      }
      try {
        const response = await searchDrug(drug, condition)
        if (active) setData(response.data)
      } catch (err) {
        if (active) setError(err?.response?.data?.detail || 'Unable to fetch search results.')
      } finally {
        if (active) setLoading(false)
      }
    }

    run()
    return () => {
      active = false
    }
  }, [drug, condition])

  const goCompare = () => {
    if (!compareDrug.trim()) return
    navigate(
      `/compare?drug1=${encodeURIComponent(drug)}&drug2=${encodeURIComponent(compareDrug.trim())}&condition=${encodeURIComponent(condition)}`
    )
  }

  return (
    <section className="mx-auto max-w-5xl px-4 py-8">
      {loading && <Spinner label="Loading search results..." />}
      {error && <p className="rounded-lg bg-red-50 p-3 text-sm text-red-600">{error}</p>}
      {!loading && !error && data && (
        <div className="space-y-6">
          <div>
            <h2 className="text-2xl font-semibold text-gray-900">
              {data.drug} for {data.condition}
            </h2>
            <p className="text-sm text-gray-500">{data.total_reviews} total reviews</p>
          </div>
          <SentimentBar distribution={data.sentiment_distribution} />
          <div className="space-y-3">
            {data.top_reviews.map((item, idx) => (
              <ReviewCard
                key={idx}
                review={item.review}
                usefulCount={item.usefulCount}
                rating={item.rating}
              />
            ))}
          </div>
          <div className="rounded-xl bg-white p-4 shadow-sm">
            <button
              type="button"
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
              onClick={() => setShowCompareInput((prev) => !prev)}
            >
              Compare with another drug
            </button>
            {showCompareInput && (
              <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                <input
                  value={compareDrug}
                  onChange={(e) => setCompareDrug(e.target.value)}
                  placeholder="Enter second drug"
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-500"
                />
                <button
                  type="button"
                  onClick={goCompare}
                  className="rounded-lg border border-indigo-600 px-4 py-2 text-sm font-medium text-indigo-600 hover:bg-indigo-50"
                >
                  Compare
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
