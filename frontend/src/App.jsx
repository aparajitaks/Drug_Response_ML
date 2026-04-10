import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import SearchResults from './pages/SearchResults'
import Compare from './pages/Compare'
import Predict from './pages/Predict'

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main className="min-h-screen bg-gray-50">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/search" element={<SearchResults />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/predict" element={<Predict />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
