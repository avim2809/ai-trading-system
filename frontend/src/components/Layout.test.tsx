import { describe, it, expect } from 'vitest'
import userEvent from '@testing-library/user-event'
import { screen } from '@testing-library/react'
import Layout from './Layout'
import { renderWithProviders } from '../test/utils'

describe('Layout — mobile nav', () => {
  it('starts with the nav drawer closed and no backdrop', () => {
    const { container } = renderWithProviders(<Layout />, { path: '/*' })
    expect(container.querySelector('aside')?.className).toContain('-translate-x-full')
    expect(container.querySelector('.bg-black\\/60')).not.toBeInTheDocument()
  })

  it('opens the nav drawer when the hamburger button is tapped', async () => {
    const user = userEvent.setup()
    const { container } = renderWithProviders(<Layout />, { path: '/*' })

    await user.click(screen.getByRole('button', { name: /open menu/i }))

    expect(container.querySelector('aside')?.className).toContain('translate-x-0')
    expect(container.querySelector('.bg-black\\/60')).toBeInTheDocument()
  })

  it('closes the nav drawer when the backdrop is clicked', async () => {
    const user = userEvent.setup()
    const { container } = renderWithProviders(<Layout />, { path: '/*' })

    await user.click(screen.getByRole('button', { name: /open menu/i }))
    expect(container.querySelector('.bg-black\\/60')).toBeInTheDocument()

    await user.click(container.querySelector('.bg-black\\/60')!)
    expect(container.querySelector('aside')?.className).toContain('-translate-x-full')
  })

  it('closes the nav drawer when a nav link is clicked', async () => {
    const user = userEvent.setup()
    const { container } = renderWithProviders(<Layout />, { path: '/*' })

    await user.click(screen.getByRole('button', { name: /open menu/i }))
    expect(container.querySelector('aside')?.className).toContain('translate-x-0')

    await user.click(screen.getByRole('link', { name: /live dashboard/i }))
    expect(container.querySelector('aside')?.className).toContain('-translate-x-full')
  })

  it('closes the drawer via the close (X) button', async () => {
    const user = userEvent.setup()
    const { container } = renderWithProviders(<Layout />, { path: '/*' })

    await user.click(screen.getByRole('button', { name: /open menu/i }))
    await user.click(screen.getByRole('button', { name: /close menu/i }))

    expect(container.querySelector('aside')?.className).toContain('-translate-x-full')
  })
})
