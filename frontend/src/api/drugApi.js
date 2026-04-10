import client from './client'

export const searchDrug = (drug, condition) =>
  client.get('/search', { params: { drug, condition } })

export const compareDrugs = (drug1, drug2, condition) =>
  client.get('/compare', { params: { drug1, drug2, condition } })

export const predictResponse = (drugName, condition, review, usefulCount) =>
  client.post('/predict', { drugName, condition, review, usefulCount })

export const getAllDrugs = () => client.get('/drugs')
