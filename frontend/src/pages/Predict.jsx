import { useState } from 'react'
import { predictResponse } from '../api/drugApi'
import Spinner from '../components/Spinner'
import PredictionCard from '../components/PredictionCard'
import ShapExplanation from '../components/ShapExplanation'

export default function Predict() {
  const [drugName, setDrugName] = useState('')
  const [condition, setCondition] = useState('')
  const [review, setReview] = useState('')
  const [usefulCount, setUsefulCount] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const onSubmit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const response = await predictResponse(drugName, condition, review, Number(usefulCount))
      setResult(response.data)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Prediction request failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
      <h1 className="text-2xl font-bold text-gray-900 sm:text-3xl">Analyze Drug Review</h1>
      <form onSubmit={onSubmit} className="mt-5 space-y-3 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <input
          type="text"
          placeholder="Drug Name"
          value={drugName}
          onChange={(e) => setDrugName(e.target.value)}
          className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm outline-none ring-indigo-500 focus:ring"
          required
        />
        <input
          type="text"
          placeholder="Condition"
          value={condition}
          onChange={(e) => setCondition(e.target.value)}
          className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm outline-none ring-indigo-500 focus:ring"
          required
        />
        <textarea
          rows={3}
          placeholder="Review"
          value={review}
          onChange={(e) => setReview(e.target.value)}
          className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm outline-none ring-indigo-500 focus:ring"
          required
        />
        <input
          type="number"
          min="0"
          value={usefulCount}
          onChange={(e) => setUsefulCount(e.target.value)}
          className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm outline-none ring-indigo-500 focus:ring"
        />
        <button
          type="submit"
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
          disabled={loading}
        >
          Analyze Review
        </button>
      </form>

      {loading ? <Spinner label="Analyzing review..." /> : null}
      {error ? <p className="mt-4 rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}

      {result ? (
        <div className="mt-6 space-y-4">
          <PredictionCard
            prediction_label={result.prediction_label}
            prediction_class={result.prediction_class}
            confidence={result.confidence}
          />
          <ShapExplanation shap_explanation={result.shap_explanation || []} />
        </div>
      ) : null}
    </section>
  )
}
