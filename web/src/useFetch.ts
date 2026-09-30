import { useEffect, useState } from 'react'

import { getJSON } from './api'

export type FetchState<T> = {
  data: T | null
  error: unknown
  loading: boolean
}

export function useFetch<T>(path: string | null): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({
    data: null,
    error: null,
    loading: Boolean(path),
  })

  useEffect(() => {
    if (!path) {
      setState({ data: null, error: null, loading: false })
      return
    }

    const controller = new AbortController()
    setState(previous => ({ ...previous, error: null, loading: true }))
    getJSON<T>(path, controller.signal)
      .then(data => setState({ data, error: null, loading: false }))
      .catch(error => {
        if (!controller.signal.aborted) setState({ data: null, error, loading: false })
      })

    return () => controller.abort()
  }, [path])

  return state
}
