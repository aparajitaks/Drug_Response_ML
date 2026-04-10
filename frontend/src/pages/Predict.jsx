import { useState } from 'react'
import { predictResponse } from '../api/drugApi'
import PredictionCard from '../components/PredictionCard'
import ShapExplanation from '../components/ShapExplanation'
import Spinner from '../components/Spinner'

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
      setError(err?.response?.data?.detail || 'Unable to analyze this review.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="mx-auto max-w-3xl px-4 py-8">
      <form onSubmit={onSubmit} className="space-y-4 rounded-xl bg-white p-5 shadow-sm">
        <h2 className="text-2xl font-semibold text-gray-900">Analyze Review</h2>
        <input
          value={drugName}
          onChange={(e) => setDrugName(e.target.value)}
          placeholder="Drug Name"
          required
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-500"
        />
        <input
          value={condition}
          onChange={(e) => setCondition(e.target.value)}
          placeholder="Condition"
          required
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-500"
        />
        <textarea
          value={review}
          onChange={(e) => setReview(e.target.value)}
          placeholder="Review"
          rows={4}
          required
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-500"
        />
        <input
          value={usefulCount}
          onChange={(e) => setUsefulCount(e.target.value)}
          type="number"
          min={0}
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          Analyze Review
        </button>
      </form>

      {loading && <Spinner label="Running prediction..." />}
      {error && <p className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-600">{error}</p>}
      {result && (
        <div className="mt-6 space-y-4">
          <PredictionCard
            prediction_label={result.prediction_label}
            prediction_class={result.prediction_class}
            confidence={result.confidence}
          />
          <ShapExplanation shap_explanation={result.shap_explanation} />
        </div>
      )}
    </section>
  )
}
