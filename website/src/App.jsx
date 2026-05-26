import Nav from './components/Nav'
import Hero from './components/Hero'
import Abstract from './components/Abstract'
import Method from './components/Method'
import Results from './components/Results'
import RetrievalDemo from './components/RetrievalDemo'
import Bibtex from './components/Bibtex'
import Footer from './components/Footer'

export default function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Abstract />
        <Method />
        <Results />
        <RetrievalDemo />
        <Bibtex />
      </main>
      <Footer />
    </>
  )
}
