import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { compareDrugs } from '../api/drugApi'
import Spinner from '../components/Spinner'
import SentimentBar from '../components/SentimentBar'
import ReviewCard from '../components/ReviewCard'

function CompareColumn({ title, data }) {
  return (
    <div className="space-y-4 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
      <SentimentBar distribution={data.sentiment_distribution} />
      <div className="space-y-3">
        {data.top_reviews.slice(0, 3).map((review, idx) => (
          <ReviewCard key={idx} review={review.review} usefulCount={review.usefulCount} rating={review.rating} />
        ))}
      </div>
    </div>
  )
}

export default function Compare() {
  const [params] = useSearchParams()
  const drug1 = params.get('drug1') || ''
  const drug2 = params.get('drug2') || ''
  const condition = params.get('condition') || ''

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!drug1 || !drug2 || !condition) return
    const load = async () => {
      setLoading(true)
      setError('')
      try {
        const response = await compareDrugs(drug1, drug2, condition)
        setData(response.data)
      } catch (err) {
        setError(err?.response?.data?.detail || err.message || 'Compare request failed.')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [drug1, drug2, condition])

  if (loading) return <Spinner label="Comparing drugs..." />

  return (
    <section className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      {error ? <p className="rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}
      {data ? (
        <>
          <h1 className="text-2xl font-bold text-gray-900 sm:text-3xl">
            Compare: {drug1} vs {drug2}
          </h1>
          <p className="mt-1 text-sm text-gray-500">Condition: {condition}</p>

          <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <CompareColumn title={data.drug1_data.drug} data={data.drug1_data} />
            <CompareColumn title={data.drug2_data.drug} data={data.drug2_data} />
          </div>

          <div className="mt-6 rounded-lg border border-indigo-200 bg-indigo-50 p-4 text-sm text-indigo-800">
            {data.simple_insight}
          </div>

          <Link
            to="/predict"
            className="mt-6 inline-block rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
          >
            Analyze a specific review
          </Link>
        </>
      ) : null}
    </section>
  )
}
