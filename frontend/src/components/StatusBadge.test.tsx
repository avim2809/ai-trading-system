import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import StatusBadge from './StatusBadge'

describe('StatusBadge', () => {
  it('renders the status text', () => {
    render(<StatusBadge status="completed" />)
    expect(screen.getByText('completed')).toBeInTheDocument()
  })

  it('shows a pulsing dot only for running status', () => {
    const { container, rerender } = render(<StatusBadge status="running" />)
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument()

    rerender(<StatusBadge status="completed" />)
    expect(container.querySelector('.animate-pulse')).not.toBeInTheDocument()
  })

  it('falls back to a neutral style for an unknown status', () => {
    render(<StatusBadge status="something_unrecognized" />)
    expect(screen.getByText('something_unrecognized')).toBeInTheDocument()
  })
})
