// custom search widget
(async function() {
    const MAX_RESULTS = 20
    let FOCUSABLE_ELEMENTS = []
    let FOCUSED_ELEMENT_INDEX = 0

    const pagefind = await import(window.MULTIDOCUMENTER_ROOT_PATH + "pagefind/pagefind.js")

    function initialize() {
        pagefind.init()
        registerSearchListener()

        document.body.addEventListener('keydown', ev => {
            if (document.activeElement === document.body && (ev.key === '/' || ev.key === 's')) {
                document.getElementById('search-input').focus()
                ev.preventDefault()
            }
        })
    }

    function registerSearchListener() {
        const input = document.getElementById('search-input')
        const suggestions = document.getElementById('search-result-container')

        async function runSearch() {
            const query = input.value

            const search = await pagefind.debouncedSearch(query, {}, 300);

            if (search) {
                buildResults(search.results)
            }
        }

        input.addEventListener('keyup', ev => {
            runSearch()
        })

        input.addEventListener('keydown', ev => {
            if (ev.key === 'ArrowDown') {
                FOCUSED_ELEMENT_INDEX = 0
                FOCUSABLE_ELEMENTS[FOCUSED_ELEMENT_INDEX].focus()
                ev.preventDefault()
                return
            } else if (ev.key === 'ArrowUp') {
                FOCUSED_ELEMENT_INDEX = FOCUSABLE_ELEMENTS.length - 1
                FOCUSABLE_ELEMENTS[FOCUSED_ELEMENT_INDEX].focus()
                ev.preventDefault()
                return
            }
        })

        suggestions.addEventListener('keydown', ev => {
            if (ev.key === 'ArrowDown') {
                FOCUSED_ELEMENT_INDEX += 1
                if (FOCUSED_ELEMENT_INDEX < FOCUSABLE_ELEMENTS.length) {
                    FOCUSABLE_ELEMENTS[FOCUSED_ELEMENT_INDEX].focus()
                } else {
                    FOCUSED_ELEMENT_INDEX = -1
                    input.focus()
                }
                ev.preventDefault()
            } else if (ev.key === 'ArrowUp') {
                FOCUSED_ELEMENT_INDEX -= 1
                if (FOCUSED_ELEMENT_INDEX >= 0) {
                    FOCUSABLE_ELEMENTS[FOCUSED_ELEMENT_INDEX].focus()
                } else {
                    FOCUSED_ELEMENT_INDEX = -1
                    input.focus()
                }
                ev.preventDefault()
            }
        })

        input.addEventListener('focus', ev => {
            runSearch()
        })
    }

    function renderResult(result) {
        const entry = document.createElement('li')
        entry.classList.add('suggestion')

        const linkContainer = document.createElement('a')
        linkContainer.classList.add('suggestion-header')
        linkContainer.setAttribute('href', result.url)

        const page = document.createElement('p')
        page.classList.add('suggestion-title')

        const pageTitle = document.createElement('span')
        pageTitle.innerText = result.title ?? result.meta.title

        page.appendChild(pageTitle)

        const excerpt = document.createElement('p')
        excerpt.classList.add('suggestion-excerpt')
        excerpt.innerHTML = result.excerpt

        linkContainer.appendChild(page)
        linkContainer.appendChild(excerpt)

        entry.appendChild(linkContainer)

        return entry
    }

    async function buildResults(results) {
        const suggestions = document.getElementById('search-result-container')

        const children = await Promise.all(results.slice(0, MAX_RESULTS - 1).map(async (r, i) => {
            const data = await r.data()

            const entry = renderResult(data)

            if (data.sub_results.length > 0) {
                const subResults = document.createElement('ol')
                subResults.classList.add('sub-suggestions')

                data.sub_results.forEach(subresult => {
                    const entry = renderResult(subresult)
                    subResults.appendChild(entry)
                })
                entry.appendChild(subResults)
            }
            
            return entry
        }))

        if (results.length > 0) {
            suggestions.classList.remove('hidden')
        } else {
            suggestions.classList.add('hidden')
        }

        
        suggestions.replaceChildren(
            ...children
        )

        FOCUSED_ELEMENT_INDEX = -1
        FOCUSABLE_ELEMENTS = [...suggestions.querySelectorAll('a')]
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize)
    } else {
        initialize()
    };
})()
