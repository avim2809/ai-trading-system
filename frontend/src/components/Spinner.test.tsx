import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import Spinner from './Spinner'

describe('Spinner', () => {
  it('renders an animated svg with the base classes', () => {
    const { container } = render(<Spinner />)
    const svg = container.querySelector('svg')
    expect(svg).toHaveClass('animate-spin', 'text-blue-400')
  })

  it('merges a custom className onto the default classes', () => {
    const { container } = render(<Spinner className="h-3.5 w-3.5" />)
    expect(container.querySelector('svg')).toHaveClass('h-3.5', 'w-3.5', 'animate-spin')
  })
})
