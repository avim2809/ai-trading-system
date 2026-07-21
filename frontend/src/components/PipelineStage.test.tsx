import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import PipelineStage from './PipelineStage'

describe('PipelineStage', () => {
  it('renders the title and children', () => {
    render(<PipelineStage title="Analysts">child content</PipelineStage>)
    expect(screen.getByText('Analysts')).toBeInTheDocument()
    expect(screen.getByText('child content')).toBeInTheDocument()
  })

  it('uses a neutral border with no status', () => {
    const { container } = render(<PipelineStage title="Risk">x</PipelineStage>)
    expect(container.querySelector('.border-slate-700')).toBeInTheDocument()
  })

  it('uses an emerald border/dot when complete', () => {
    const { container } = render(<PipelineStage title="Risk" status="complete">x</PipelineStage>)
    expect(container.querySelector('.border-emerald-600')).toBeInTheDocument()
    expect(container.querySelector('.bg-emerald-500')).toBeInTheDocument()
  })
})
