import { describe, it, expect, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { renderWithProviders } from '../test/utils'
import LiveConfig from './LiveConfig'

describe('LiveConfig', () => {
  it('renders the RAG collections table with the real nested shape', async () => {
    renderWithProviders(<LiveConfig />)
    await waitFor(() => expect(screen.getByText('sec_filings')).toBeInTheDocument())
    expect(screen.getByText('4,205')).toBeInTheDocument()
  })

  it('does not crash if /llm/rag/stats ever regresses to a flat shape', async () => {
    // Regression test for the actual incident: the backend used to return
    // {"sec_filings": 4205, "_total": 4205} with no "collections" key at
    // all, and Object.keys(ragStats.collections) threw, blanking the whole
    // page. The page now uses optional chaining so a shape regression like
    // this just hides the RAG table instead of crashing everything else.
    server.use(
      http.get('http://localhost/api/llm/rag/stats', () =>
        HttpResponse.json({ sec_filings: 4205, _total: 4205 })),
    )
    renderWithProviders(<LiveConfig />)
    // The rest of the page (which doesn't depend on rag/stats) must still
    // render — proof the page as a whole survived the bad response.
    await waitFor(() => expect(screen.getByText('Live Configuration')).toBeInTheDocument())
    expect(await screen.findByText('Knowledge Base (RAG)')).toBeInTheDocument()
    expect(screen.queryByText('sec_filings')).not.toBeInTheDocument()
  })

  it('filters the provider dropdown to configured providers only', async () => {
    renderWithProviders(<LiveConfig />)
    await waitFor(() => expect(screen.getByText('Provider & Model')).toBeInTheDocument())
    const providerSelect = screen.getByDisplayValue('groq/llama-3.3-70b-versatile') // active model preselected
    void providerSelect
    // mockLLMProviders: groq is configured:true, anthropic is configured:false.
    const select = screen.getAllByRole('combobox').find((el) =>
      Array.from(el.querySelectorAll('option')).some((o) => o.textContent === 'groq'))!
    const optionTexts = Array.from(select.querySelectorAll('option')).map((o) => o.textContent)
    expect(optionTexts).toContain('groq')
    expect(optionTexts).not.toContain('anthropic')
  })

  it('deletes a RAG collection after confirmation', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    let deleteCalled = false
    server.use(
      http.delete('http://localhost/api/llm/rag/:collection', () => {
        deleteCalled = true
        return HttpResponse.json({ status: 'deleted' })
      }),
    )
    const user = userEvent.setup()
    renderWithProviders(<LiveConfig />)
    await waitFor(() => expect(screen.getByText('sec_filings')).toBeInTheDocument())

    await user.click(screen.getAllByText('Delete')[0]!)
    expect(confirmSpy).toHaveBeenCalled()
    await waitFor(() => expect(deleteCalled).toBe(true))
    confirmSpy.mockRestore()
  })

  it('save button submits both live config and LLM config updates', async () => {
    let liveConfigSaved = false
    let llmConfigSaved = false
    server.use(
      http.put('http://localhost/api/live/config', () => { liveConfigSaved = true; return HttpResponse.json({ status: 'updated' }) }),
      http.put('http://localhost/api/llm/config', () => { llmConfigSaved = true; return HttpResponse.json({ status: 'updated' }) }),
    )
    const user = userEvent.setup()
    renderWithProviders(<LiveConfig />)
    await waitFor(() => expect(screen.getByText('Save Configuration')).toBeInTheDocument())
    await user.click(screen.getByText('Save Configuration'))
    await waitFor(() => expect(liveConfigSaved).toBe(true))
    expect(llmConfigSaved).toBe(true)
  })
})
