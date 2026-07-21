import type { ReactElement, ReactNode } from 'react'
import { render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Routes, Route } from 'react-router-dom'

function newTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: Infinity },
      mutations: { retry: false },
    },
  })
}

/** Wraps a page/component with the same providers App.tsx sets up. */
export function renderWithProviders(
  ui: ReactElement,
  { route = '/', path = '/' }: { route?: string; path?: string } = {},
) {
  const queryClient = newTestQueryClient()
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[route]}>
          <Routes>
            <Route path={path} element={children} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    )
  }
  return render(ui, { wrapper: Wrapper })
}

export * from '@testing-library/react'
