import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { compareDrugs } from '../api/drugApi'
import ReviewCard from '../components/ReviewCard'
import SentimentBar from '../components/SentimentBar'
import Spinner from '../components/Spinner'

function DrugColumn({ title, data }) {
  return (
    <div className="space-y-4 rounded-xl bg-white p-4 shadow-sm">
      <h3 className="text-xl font-semibold text-gray-900">{title}</h3>
      <SentimentBar distribution={data?.sentiment_distribution} />
      <div className="space-y-3">
        {(data?.top_reviews || []).slice(0, 3).map((review, idx) => (
          <ReviewCard
            key={idx}
            review={review.review}
            usefulCount={review.usefulCount}
            rating={review.rating}
          />
        ))}
      </div>
    </div>
  )
}

export default function Compare() {
  const [params] = useSearchParams()
  const drug1 = (params.get('drug1') || '').trim()
  const drug2 = (params.get('drug2') || '').trim()
  const condition = (params.get('condition') || '').trim()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [data, setData] = useState(null)

  useEffect(() => {
    let active = true
    const run = async () => {
      setLoading(true)
      setError('')
      if (!drug1 || !drug2 || !condition) {
        setError('Missing compare query parameters. Provide drug1, drug2, and condition.')
        setLoading(false)
        return
      }
      try {
        const response = await compareDrugs(drug1, drug2, condition)
        if (active) setData(response.data)
      } catch (err) {
        if (active) setError(err?.response?.data?.detail || 'Unable to compare drugs.')
      } finally {
        if (active) setLoading(false)
      }
    }
    run()
    return () => {
      active = false
    }
  }, [drug1, drug2, condition])

  return (
    <section className="mx-auto max-w-6xl px-4 py-8">
      {loading && <Spinner label="Comparing drugs..." />}
      {error && <p className="rounded-lg bg-red-50 p-3 text-sm text-red-600">{error}</p>}
      {!loading && !error && data && (
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <DrugColumn title={data.drug1_data.drug} data={data.drug1_data} />
            <DrugColumn title={data.drug2_data.drug} data={data.drug2_data} />
          </div>
          <div className="rounded-xl border border-indigo-100 bg-indigo-50 p-4 text-sm text-indigo-900">
            {data.simple_insight}
          </div>
          <Link
            to="/predict"
            className="inline-flex rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
          >
            Analyze a specific review
          </Link>
        </div>
      )}
    </section>
  )
}
