"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type ResearchRecord = {
  record_id: string;
  published_at: string;
  source_type: string;
  source_name: string;
  headline: string;
  content: string;
  company: string;
  ticker: string;
  sector: string;
  industry: string;
  country: string;
  tags: string;
  source_path: string;
  language: string;
  [key: string]: string;
};

type Company = {
  company: string;
  ticker: string;
  aliases: string;
  sector: string;
  industry: string;
  geography: string;
  description: string;
  [key: string]: string;
};

type WatchlistEntry = { ticker: string; company: string; [key: string]: string };
type Tab = "feed" | "companies" | "saved" | "watchlist";
type Theme = "dark" | "light";

const DATA_ROOT = "/data/Aequitas_Research_Intelligence_Assignment_Data";

function parseCSV(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  const input = text.replace(/^\uFEFF/, "");

  for (let i = 0; i < input.length; i++) {
    const char = input[i];
    const next = input[i + 1];

    if (char === '"') {
      if (quoted && next === '"') {
        cell += '"';
        i++;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i++;
      row.push(cell);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  if (cell.length || row.length) {
    row.push(cell);
    if (row.some((value) => value.trim())) rows.push(row);
  }

  if (!rows.length) return [];
  const headers = rows[0].map((header) => header.trim());
  return rows.slice(1).map((values) => {
    const item: Record<string, string> = {};
    headers.forEach((header, index) => (item[header] = (values[index] ?? "").trim()));
    return item;
  });
}

function clean(value?: string) {
  return value?.trim() || "";
}

function unique(values: string[]) {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function tagsOf(value: string) {
  return value ? value.split(/[;,|]/).map((tag) => tag.trim()).filter(Boolean) : [];
}

function dateValue(value: string) {
  const parsed = new Date((value || "").replace(" ", "T")).getTime();
  return Number.isNaN(parsed) ? 0 : parsed;
}

function formatDate(value: string, withYear = true) {
  if (!value) return "Date unavailable";
  const date = new Date(value.replace(" ", "T"));
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    ...(withYear ? { year: "numeric" as const } : {}),
  }).format(date);
}

function priorityFor(record: ResearchRecord) {
  // Transparent heuristic, not an AI or investment recommendation.
  const text = `${record.headline} ${record.content} ${record.tags} ${record.source_type}`.toLowerCase();
  let score = 35;
  if (/results|earnings|guidance|profit|revenue|margin|quarter/.test(text)) score += 18;
  if (/regulatory|order|approval|investigation|penalty|court| sebi|rbi/.test(text)) score += 20;
  if (/acquisition|merger|stake|buyback|fundrais|default|downgrade|upgrade/.test(text)) score += 17;
  if (/exchange filing|company announcement|rating agency/.test(text)) score += 8;
  if (/newswire|press release/.test((record.source_type || "").toLowerCase())) score += 3;
  return Math.min(98, score);
}

function priorityLabel(score: number) {
  if (score >= 75) return "High";
  if (score >= 58) return "Medium";
  return "Routine";
}

function highlight(text: string, query: string, theme: Theme) {
  if (!query.trim() || !text) return text;
  const escaped = query.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pieces = text.split(new RegExp(`(${escaped})`, "ig"));
  return pieces.map((piece, index) =>
    piece.toLowerCase() === query.trim().toLowerCase() ? (
      <mark
        key={`${piece}-${index}`}
        className={theme === "dark" ? "rounded-sm bg-[#75612a]/60 text-[#fff0b8]" : "rounded-sm bg-[#f4e6ad] text-[#433510]"}
      >
        {piece}
      </mark>
    ) : (
      piece
    )
  );
}



export default function Home() {
  const [records, setRecords] = useState<ResearchRecord[]>([]);
  const [companies, setCompanies] = useState<Company[]>([]);
  const [initialWatchlist, setInitialWatchlist] = useState<WatchlistEntry[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>("feed");
  const [selectedRecord, setSelectedRecord] = useState<ResearchRecord | null>(null);
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);
  const [query, setQuery] = useState("");
  const [sourceFilter, setSourceFilter] = useState("All sources");
  const [sectorFilter, setSectorFilter] = useState("All sectors");
  const [companyFilter, setCompanyFilter] = useState("All companies");
  const [priorityFilter, setPriorityFilter] = useState("All priorities");
  const [dateFilter, setDateFilter] = useState("Any time");
  const [sortOrder, setSortOrder] = useState("Newest first");
  const [savedIds, setSavedIds] = useState<string[]>([]);
  const [watchlistTickers, setWatchlistTickers] = useState<string[]>([]);
  const [documentText, setDocumentText] = useState("");
  const [documentLoading, setDocumentLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [theme, setTheme] = useState<Theme>("light");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searches, setSearches] = useState<string[]>([]);
  const [activeTag, setActiveTag] = useState("");
  const [showAllTags, setShowAllTags] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const rowRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("aequitas-saved-records");
      const watchlist = localStorage.getItem("aequitas-watchlist");
      const storedTheme = localStorage.getItem("aequitas-theme");
      const storedSearches = localStorage.getItem("aequitas-saved-searches");
      if (saved) setSavedIds(JSON.parse(saved));
      if (watchlist) setWatchlistTickers(JSON.parse(watchlist));
      if (storedTheme === "light" || storedTheme === "dark") setTheme(storedTheme);
      if (storedSearches) setSearches(JSON.parse(storedSearches));
    } catch {
      // Invalid local storage is ignored.
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadDataset() {
      setLoading(true);
      setLoadError("");
      try {
        const [feedRes, companyRes, watchRes] = await Promise.all([
          fetch(`${DATA_ROOT}/research_feed.csv`),
          fetch(`${DATA_ROOT}/company_master.csv`),
          fetch(`${DATA_ROOT}/watchlist.csv`),
        ]);
        if (!feedRes.ok || !companyRes.ok || !watchRes.ok) {
          throw new Error(`Dataset files not found. Check public/data/Aequitas_Research_Intelligence_Assignment_Data/ and confirm the CSV filenames.`);
        }
        const [feedText, companyText, watchText] = await Promise.all([
          feedRes.text(), companyRes.text(), watchRes.text(),
        ]);
        const parsedRecords = parseCSV(feedText) as unknown as ResearchRecord[];
        const parsedCompanies = parseCSV(companyText) as unknown as Company[];
        const parsedWatchlist = parseCSV(watchText) as unknown as WatchlistEntry[];
        if (!parsedRecords.length) throw new Error("The research feed loaded, but it contains no records.");
        if (!cancelled) {
          setRecords(parsedRecords);
          setCompanies(parsedCompanies);
          setInitialWatchlist(parsedWatchlist);
          setSelectedRecord((current) => current ?? [...parsedRecords].sort((a, b) => priorityFor(b) - priorityFor(a) || dateValue(b.published_at) - dateValue(a.published_at))[0] ?? null);
          setWatchlistTickers((current) => current.length ? current : parsedWatchlist.map((item) => clean(item.ticker)).filter(Boolean));
        }
      } catch (error) {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : "Unable to load assignment data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadDataset();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!selectedRecord) {
      setDocumentText("");
      setDocumentLoading(false);
      return;
    }
    let cancelled = false;
    async function loadDocument() {
      setDocumentText("");
      setDocumentLoading(true);
      try {
        const path = clean(selectedRecord?.source_path);
        if (!path) throw new Error("No source document path is available.");
        const safePath = path.split("/").map((part) => encodeURIComponent(part)).join("/");
        const response = await fetch(`${DATA_ROOT}/${safePath}`);
        if (!response.ok) throw new Error("Original text document could not be found.");
        const text = await response.text();
        if (!cancelled) setDocumentText(text);
      } catch (error) {
        if (!cancelled) setDocumentText(error instanceof Error ? error.message : "Unable to load source document.");
      } finally {
        if (!cancelled) setDocumentLoading(false);
      }
    }
    loadDocument();
    return () => { cancelled = true; };
  }, [selectedRecord]);

  const persistSaved = useCallback((next: string[]) => {
    setSavedIds(next);
    try { localStorage.setItem("aequitas-saved-records", JSON.stringify(next)); } catch {}
  }, []);

  const persistWatchlist = useCallback((next: string[]) => {
    setWatchlistTickers(next);
    try { localStorage.setItem("aequitas-watchlist", JSON.stringify(next)); } catch {}
  }, []);

  const toggleSaved = useCallback((id: string) => {
    persistSaved(savedIds.includes(id) ? savedIds.filter((item) => item !== id) : [...savedIds, id]);
  }, [savedIds, persistSaved]);

  const toggleWatchlist = useCallback((ticker: string) => {
    if (!ticker) return;
    persistWatchlist(watchlistTickers.includes(ticker) ? watchlistTickers.filter((item) => item !== ticker) : [...watchlistTickers, ticker]);
  }, [watchlistTickers, persistWatchlist]);

  const sourceTypes = useMemo(() => unique(records.map((record) => clean(record.source_type))), [records]);
  const sectors = useMemo(() => unique(records.map((record) => clean(record.sector))), [records]);
  const feedCompanies = useMemo(() => unique(records.map((record) => clean(record.company)).filter(Boolean)), [records]);
  const allTags = useMemo(() => unique(records.flatMap((record) => tagsOf(record.tags))), [records]);

  const companyMap = useMemo(() => {
    const map = new Map<string, Company>();
    companies.forEach((company) => { if (company.ticker) map.set(company.ticker.toUpperCase(), company); });
    return map;
  }, [companies]);

  const filteredRecords = useMemo(() => {
    let result = [...records];
    if (activeTab === "saved") result = result.filter((record) => savedIds.includes(record.record_id));
    if (activeTab === "watchlist") result = result.filter((record) => record.ticker && watchlistTickers.includes(record.ticker.toUpperCase()));
    if (companyFilter !== "All companies") result = result.filter((record) => record.company === companyFilter);
    if (sourceFilter !== "All sources") result = result.filter((record) => record.source_type === sourceFilter);
    if (sectorFilter !== "All sectors") result = result.filter((record) => record.sector === sectorFilter);
    if (priorityFilter !== "All priorities") result = result.filter((record) => priorityLabel(priorityFor(record)) === priorityFilter);
    if (dateFilter !== "Any time") {
      const now = Date.now();
      const days = dateFilter === "24 hours" ? 1 : dateFilter === "7 days" ? 7 : dateFilter === "30 days" ? 30 : 365;
      result = result.filter((record) => dateValue(record.published_at) >= now - days * 86400000);
    }
    if (activeTag) result = result.filter((record) => tagsOf(record.tags).some((tag) => tag.toLowerCase() === activeTag.toLowerCase()));
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      result = result.filter((record) => [
        record.record_id, record.headline, record.content, record.company, record.ticker,
        record.sector, record.industry, record.source_type, record.source_name,
        record.country, record.tags,
      ].join(" ").toLowerCase().includes(q));
    }
    result.sort((a, b) => sortOrder === "Priority" ? priorityFor(b) - priorityFor(a) : sortOrder === "Oldest first" ? dateValue(a.published_at) - dateValue(b.published_at) : dateValue(b.published_at) - dateValue(a.published_at));
    return result;
  }, [records, activeTab, savedIds, watchlistTickers, companyFilter, sourceFilter, sectorFilter, priorityFilter, dateFilter, activeTag, query, sortOrder]);

  const displayedCompanies = useMemo(() => {
    const q = query.trim().toLowerCase();
    return companies.filter((company) => !q || [company.company, company.ticker, company.aliases, company.sector, company.industry, company.geography, company.description].join(" ").toLowerCase().includes(q));
  }, [companies, query]);

  const companyRecords = useMemo(() => {
    if (!selectedCompany) return [];
    return records.filter((record) => (record.ticker && record.ticker.toUpperCase() === selectedCompany.ticker.toUpperCase()) || record.company === selectedCompany.company)
      .sort((a, b) => dateValue(b.published_at) - dateValue(a.published_at));
  }, [records, selectedCompany]);

  const topPriorities = useMemo(() => {
    const sorted = [...filteredRecords].filter((record) => priorityFor(record) >= 58).sort((a, b) => priorityFor(b) - priorityFor(a) || dateValue(b.published_at) - dateValue(a.published_at));
    const seen = new Set<string>();
    return sorted.filter((record) => {
      const key = (record.ticker || record.company || record.record_id).trim().toLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    }).slice(0, 3);
  }, [filteredRecords]);

  const clearFilters = () => {
    setQuery("");
    setSourceFilter("All sources");
    setSectorFilter("All sectors");
    setCompanyFilter("All companies");
    setPriorityFilter("All priorities");
    setDateFilter("Any time");
    setActiveTag("");
  };

  const activeFilters = [
    ...(query ? [{ label: `Search: ${query}`, clear: () => setQuery("") }] : []),
    ...(sourceFilter !== "All sources" ? [{ label: sourceFilter, clear: () => setSourceFilter("All sources") }] : []),
    ...(sectorFilter !== "All sectors" ? [{ label: sectorFilter, clear: () => setSectorFilter("All sectors") }] : []),
    ...(companyFilter !== "All companies" ? [{ label: companyFilter, clear: () => setCompanyFilter("All companies") }] : []),
    ...(priorityFilter !== "All priorities" ? [{ label: `${priorityFilter} priority`, clear: () => setPriorityFilter("All priorities") }] : []),
    ...(dateFilter !== "Any time" ? [{ label: dateFilter, clear: () => setDateFilter("Any time") }] : []),
    ...(activeTag ? [{ label: `#${activeTag}`, clear: () => setActiveTag("") }] : []),
  ];

  const openRecord = (record: ResearchRecord) => {
    setSelectedRecord(record);
    setSelectedCompany(null);
    setSidebarOpen(false);
  };

  const openCompany = (company: Company) => {
    setSelectedCompany(company);
    setSelectedRecord(null);
    setActiveTab("companies");
    setSidebarOpen(false);
  };

  const runSavedSearch = (savedSearch: string) => {
    setQuery(savedSearch);
    setActiveTab("feed");
    setSidebarOpen(false);
  };

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing = target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
      if (event.key === "/" && !typing) {
        event.preventDefault();
        searchRef.current?.focus();
      } else if (event.key === "Escape") {
        if (document.activeElement === searchRef.current) searchRef.current?.blur();
        else { setSelectedRecord(null); setSelectedCompany(null); setSidebarOpen(false); }
      } else if (!typing && activeTab !== "companies" && (event.key === "j" || event.key === "k")) {
        event.preventDefault();
        setActiveIndex((current) => {
          const next = Math.max(0, Math.min(filteredRecords.length - 1, current + (event.key === "j" ? 1 : -1)));
          const record = filteredRecords[next];
          if (record) rowRefs.current[record.record_id]?.focus();
          return next;
        });
      } else if (!typing && event.key === "Enter" && filteredRecords[activeIndex]) {
        openRecord(filteredRecords[activeIndex]);
      } else if (!typing && event.key.toLowerCase() === "b" && filteredRecords[activeIndex]) {
        toggleSaved(filteredRecords[activeIndex].record_id);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeTab, filteredRecords, activeIndex, toggleSaved]);

  const isLight = theme === "light";
  const shell = isLight ? "bg-[#FAF9F6] text-[#1A1A1A]" : "bg-[#191918] text-[#F2F0EB]";
  const sidebar = isLight ? "bg-[#F4F2ED] border-[#E7E5E0]" : "bg-[#201F1D] border-[#373530]";
  const panel = isLight ? "bg-[#FFFFFF] border-[#E7E5E0]" : "bg-[#201F1D] border-[#373530]";
  const border = isLight ? "border-[#E7E5E0]" : "border-[#373530]";
  const muted = isLight ? "text-[#55524D]" : "text-[#B7B2A9]";
  const hover = isLight ? "hover:bg-[#F2F1ED]" : "hover:bg-[#2A2926]";

  const navItems: { id: Tab; label: string; count: number; glyph: string }[] = [
    { id: "feed", label: "All research", count: records.length, glyph: "▤" },
    { id: "companies", label: "Companies", count: companies.length, glyph: "▦" },
    { id: "saved", label: "Read later", count: savedIds.length, glyph: "☆" },
    { id: "watchlist", label: "Watchlist", count: watchlistTickers.length, glyph: "◎" },
  ];

  return (
    <main className={`min-h-screen font-sans ${shell}`} style={{ fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' }}>
      <div className="flex min-h-screen">
        {sidebarOpen && <button aria-label="Close navigation" onClick={() => setSidebarOpen(false)} className="fixed inset-0 z-40 bg-black/40 lg:hidden" />}
        <aside className={`${sidebar} ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} fixed inset-y-0 left-0 z-50 flex w-[248px] shrink-0 flex-col overflow-y-auto border-r transition-transform duration-200 lg:sticky lg:top-0 lg:h-screen lg:translate-x-0`}>
          <div className={`flex h-[62px] items-center gap-3 border-b px-5 ${border}`}>
            <div className={`flex h-8 w-8 items-center justify-center rounded-lg border text-sm font-semibold ${isLight ? "border-[#E7E5E0] bg-white" : "border-[#3b4147] bg-[#202428]"}`}>A</div>
            <div className="min-w-0">
              <div className="text-[12px] font-semibold tracking-[0.12em]">AEQUITAS</div>
              <div className={`mt-0.5 text-[10px] ${muted}`}>Research intelligence</div>
            </div>
            <button onClick={() => setSidebarOpen(false)} className={`ml-auto rounded-md px-2 py-1 text-sm lg:hidden ${muted}`}>×</button>
          </div>

          <div className="px-3 pt-4">
            <div className={`mb-2 px-2 text-[12px] font-medium ${muted}`}>Workspace</div>
            {navItems.map((item) => (
              <button key={item.id} onClick={() => { setActiveTab(item.id); setSelectedRecord(null); setSelectedCompany(null); setSidebarOpen(false); }} className={`mb-1 flex h-9 w-full items-center gap-3 rounded-md px-2.5 text-left text-[12px] transition-colors ${activeTab === item.id ? (isLight ? "bg-[#EAF0FF] text-[#1F4FD8]" : "bg-[#30302D] text-white") : `${muted} ${hover}`}`}>
                <span className="w-4 text-center text-[15px]">{item.glyph}</span>
                <span className="flex-1">{item.label}</span>
                <span className={`font-mono text-[10px] ${activeTab === item.id ? "" : "opacity-70"}`}>{item.count}</span>
              </button>
            ))}
          </div>

          <div className={`mx-4 my-4 border-t ${border}`} />
          <div className="px-3">
            <div className={`mb-2 flex items-center justify-between px-2 text-[12px] font-medium ${muted}`}><span>Saved searches</span><span>⌘</span></div>
            {searches.length === 0 ? (
              <div className={`px-2 py-2 text-[11px] leading-5 ${muted}`}>Save a search to keep a shortcut here.</div>
            ) : searches.map((savedSearch) => (
              <button key={savedSearch} onClick={() => runSavedSearch(savedSearch)} className={`flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-[11px] ${muted} ${hover}`}><span>⌕</span><span className="truncate">{savedSearch}</span></button>
            ))}
            {query.trim() && !searches.includes(query.trim()) && (
              <button onClick={() => {
                const next = [query.trim(), ...searches].slice(0, 8);
                setSearches(next);
                try { localStorage.setItem("aequitas-saved-searches", JSON.stringify(next)); } catch {}
              }} className={`mt-1 w-full rounded-md px-2 py-2 text-left text-[10px] ${muted} ${hover}`}>＋ Save current search</button>
            )}
          </div>

          <div className={`mx-4 my-4 border-t ${border}`} />
          <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-4">
            <div className={`mb-2 px-2 text-[12px] font-medium ${muted}`}>Categories</div>
            {sourceTypes.slice(0, 8).map((source) => (
              <button key={source} onClick={() => { setSourceFilter(sourceFilter === source ? "All sources" : source); setActiveTab("feed"); }} className={`flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-[11px] ${sourceFilter === source ? (isLight ? "bg-[#EAF0FF] text-[#1F4FD8]" : "bg-[#30302D] text-white") : `${muted} ${hover}`}`}><span className="text-[10px]">◇</span><span className="flex-1 truncate">{source}</span><span className="font-mono text-[9px] opacity-70">{records.filter((record) => record.source_type === source).length}</span></button>
            ))}
            <div className={`mb-2 mt-5 px-2 text-[12px] font-medium ${muted}`}>Tags</div>
            {allTags.slice(0, showAllTags ? allTags.length : 8).map((tag) => (
              <button key={tag} onClick={() => { setActiveTag(activeTag === tag ? "" : tag); setActiveTab("feed"); }} className={`flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-[11px] ${activeTag === tag ? (isLight ? "bg-[#EAF0FF] text-[#1F4FD8]" : "bg-[#30302D] text-white") : `${muted} ${hover}`}`}><span className="text-[10px]">#</span><span className="flex-1 truncate">{tag}</span><span className="font-mono text-[9px] opacity-70">{records.filter((record) => tagsOf(record.tags).includes(tag)).length}</span></button>
            ))}
            {allTags.length > 8 && <button onClick={() => setShowAllTags((value) => !value)} className={`px-2 py-2 text-[11px] text-[#1F4FD8] hover:underline`}>{showAllTags ? "Show fewer" : "Show more"}</button>}
          </div>

          <div className={`border-t p-3 ${border}`}>
            <button onClick={() => { const next = isLight ? "dark" : "light"; setTheme(next); try { localStorage.setItem("aequitas-theme", next); } catch {} }} className={`flex h-9 w-full items-center gap-3 rounded-md px-2.5 text-left text-[11px] ${muted} ${hover}`}><span>{isLight ? "☾" : "☼"}</span><span className="flex-1">{isLight ? "Dark appearance" : "Light appearance"}</span><span className={`h-4 w-7 rounded-full p-0.5 ${isLight ? "bg-[#D6D2CA]" : "bg-[#444b52]"}`}><span className={`block h-3 w-3 rounded-full bg-white transition-transform ${isLight ? "translate-x-0" : "translate-x-3"}`} /></span></button>
            <div className={`px-2 pt-2 text-[9px] ${muted}`}>Fictionalised assignment data</div>
          </div>
        </aside>

        <section className="flex min-w-0 flex-1 flex-col">
          <header className={`sticky top-0 z-20 flex min-h-[62px] items-center gap-3 border-b px-3 sm:px-5 ${panel}`}>
            <button onClick={() => setSidebarOpen(true)} className={`rounded-md px-2 py-1 text-lg lg:hidden ${muted}`} aria-label="Open navigation">☰</button>
            <div className={`hidden text-[11px] sm:block ${muted}`}>Research</div>
            <div className={`hidden h-4 border-l sm:block ${border}`} />
            <div className={`flex h-9 min-w-0 flex-1 items-center gap-2 rounded-md border px-3 ${isLight ? "border-[#E7E5E0] bg-[#FFFFFF]" : "border-[#E7E5E0] bg-[#111315]"}`}>
              <span className={`text-[17px] ${muted}`}>⌕</span>
              <input ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search research, companies, tickers…  /" className="min-w-0 flex-1 bg-transparent text-[12px] outline-none placeholder:text-[#777e86]" />
              {query && <button onClick={() => setQuery("")} className={`text-[13px] text-[#6B6862]`} aria-label="Clear search">×</button>}
              <kbd className={`hidden rounded border px-1.5 py-0.5 text-[9px] sm:inline ${border} ${muted}`}>/</kbd>
            </div>
            <button onClick={() => { if (query.trim() && !searches.includes(query.trim())) { const next = [query.trim(), ...searches].slice(0, 8); setSearches(next); try { localStorage.setItem("aequitas-saved-searches", JSON.stringify(next)); } catch {} } }} title="Save current search" className={`hidden h-9 rounded-md border px-3 text-[11px] sm:block ${border} ${muted} ${hover}`}>Save search</button>
            <div className="relative">
              <button onClick={() => setHelpOpen((open) => !open)} title="Keyboard shortcuts" aria-label="Keyboard shortcuts" className={`h-9 w-9 rounded-md border text-[13px] font-semibold ${border} ${muted} ${hover}`}>?</button>
              {helpOpen && <div className={`absolute right-0 top-11 z-50 w-56 rounded-md border p-3 shadow-lg ${panel}`}>
                <div className="mb-2 text-[12px] font-semibold">Keyboard shortcuts</div>
                <Shortcut keys="/" action="Focus search" muted={muted} />
                <Shortcut keys="j / k" action="Move through results" muted={muted} />
                <Shortcut keys="Enter" action="Open selected result" muted={muted} />
                <Shortcut keys="b" action="Bookmark result" muted={muted} />
                <Shortcut keys="Esc" action="Close detail pane" muted={muted} />
              </div>}
            </div>
            <button onClick={() => { const next = isLight ? "dark" : "light"; setTheme(next); try { localStorage.setItem("aequitas-theme", next); } catch {} }} title="Toggle theme" className={`h-9 w-9 rounded-md border text-[14px] ${border} ${muted} ${hover}`}>{isLight ? "☾" : "☼"}</button>
          </header>

          <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_360px]">
            <div className="min-w-0">
              <div className={`border-b px-4 py-3 sm:px-6 ${border}`}>
                {!selectedCompany && activeTab !== "companies" && (
                  <div className="flex min-w-0 items-center gap-2 whitespace-nowrap">
                    <div className="flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto">
                      <FilterSelect label="Date" value={dateFilter} defaultValue="Any time" onChange={setDateFilter} options={["Any time", "24 hours", "7 days", "30 days", "365 days"]} isLight={isLight} />
                      <FilterSelect label="Category" value={sourceFilter} defaultValue="All sources" onChange={setSourceFilter} options={["All sources", ...sourceTypes]} isLight={isLight} />
                      <FilterSelect label="Sector" value={sectorFilter} defaultValue="All sectors" onChange={setSectorFilter} options={["All sectors", ...sectors]} isLight={isLight} />
                      <FilterSelect label="Company" value={companyFilter} defaultValue="All companies" onChange={setCompanyFilter} options={["All companies", ...feedCompanies]} isLight={isLight} />
                      <FilterSelect label="Priority" value={priorityFilter} defaultValue="All priorities" onChange={setPriorityFilter} options={["All priorities", "High", "Medium", "Routine"]} isLight={isLight} />
                    </div>
                    <span className={`shrink-0 text-[13px] text-[#6B6862]`}>{loading ? "Loading…" : loadError ? "Data unavailable" : `${filteredRecords.length} results`}</span>
                    <select aria-label="Sort results" value={sortOrder} onChange={(event) => setSortOrder(event.target.value)} className={`h-8 max-w-[110px] shrink-0 border-0 bg-transparent text-[13px] outline-none ${isLight ? "text-[#55524D]" : "text-[#B7B2A9]"}`}><option>Newest first</option><option>Oldest first</option><option>Priority</option></select>
                  </div>
                )}
                {(selectedCompany || activeTab === "companies") && <div className="text-[13px] font-medium">{selectedCompany ? selectedCompany.company : "Companies"}</div>}
                {activeFilters.length > 0 && !selectedCompany && activeTab !== "companies" && (
                  <div className="mt-2 flex flex-wrap items-center gap-1.5">
                    {activeFilters.map((filter) => <button key={filter.label} onClick={filter.clear} className={`inline-flex h-6 items-center gap-1 rounded-full border px-2 text-[11px] ${isLight ? "border-[#E7E5E0] bg-[#F2F1ED] text-[#55524D]" : "border-[#3b4248] bg-[#22272b] text-[#c3c9ce]"}`}>{filter.label}<span className="opacity-60">×</span></button>)}
                    <button onClick={clearFilters} className={`px-2 py-1 text-[11px] underline underline-offset-2 ${muted}`}>Clear all</button>
                  </div>
                )}
              </div>

              {loading ? (
                <div className="p-8">
                  <div className={`mb-3 h-4 w-40 animate-pulse rounded ${isLight ? "bg-[#E7E5E0]" : "bg-[#272c30]"}`} />
                  {[1, 2, 3, 4, 5].map((item) => <div key={item} className={`mb-2 h-[78px] animate-pulse rounded-md border ${border} ${isLight ? "bg-white" : "bg-[#FFFFFF]"}`} />)}
                  <p className={`mt-4 text-[11px] ${muted}`}>Loading research feed and company reference data…</p>
                </div>
              ) : loadError ? (
                <div className="m-6 rounded-lg border border-[#b65f5f]/40 bg-[#7b2727]/10 p-5">
                  <div className="text-[13px] font-medium">Dataset could not be loaded</div>
                  <p className={`mt-2 text-[11px] leading-5 ${muted}`}>{loadError}</p>
                  <button onClick={() => window.location.reload()} className={`mt-4 rounded-md border px-3 py-2 text-[11px] ${border}`}>Retry</button>
                </div>
              ) : selectedCompany ? (
                <div>
                  <div className={`grid grid-cols-2 border-b sm:grid-cols-4 ${border}`}>
                    <CompanyFact label="Ticker" value={selectedCompany.ticker} isLight={isLight} />
                    <CompanyFact label="Sector" value={selectedCompany.sector} isLight={isLight} />
                    <CompanyFact label="Industry" value={selectedCompany.industry} isLight={isLight} />
                    <CompanyFact label="Geography" value={selectedCompany.geography} isLight={isLight} />
                  </div>
                  <div className={`flex items-center justify-between border-b px-4 py-3 sm:px-6 ${border}`}><span className="text-[11px] font-medium">Research timeline</span><button onClick={() => toggleWatchlist(selectedCompany.ticker)} className={`rounded-md border px-2.5 py-1.5 text-[10px] ${border}`}>{watchlistTickers.includes(selectedCompany.ticker) ? "✓ In watchlist" : "+ Add to watchlist"}</button></div>
                  {companyRecords.length ? companyRecords.map((record) => <ResearchRow key={record.record_id} record={record} query={query} theme={theme} isSaved={savedIds.includes(record.record_id)} onOpen={() => openRecord(record)} onToggleSave={() => toggleSaved(record.record_id)} isLight={isLight} />) : <EmptyState title="No related research" message="No records in the supplied dataset match this company." isLight={isLight} />}
                </div>
              ) : activeTab === "companies" ? (
                <div>
                  <div className={`flex items-center justify-between border-b px-4 py-3 sm:px-6 ${border}`}><span className={`text-[10px] uppercase tracking-[0.1em] ${muted}`}>Company reference universe</span><span className={`text-[10px] ${muted}`}>{displayedCompanies.length} companies</span></div>
                  {displayedCompanies.length ? displayedCompanies.map((company) => {
                    const count = records.filter((record) => record.ticker?.toUpperCase() === company.ticker.toUpperCase() || record.company === company.company).length;
                    return <button key={company.ticker || company.company} onClick={() => openCompany(company)} className={`grid w-full grid-cols-[minmax(0,1fr)_76px] gap-3 border-b px-4 py-3 text-left transition-colors sm:grid-cols-[minmax(0,1fr)_100px_120px_75px] sm:px-6 ${border} ${hover}`}><span className="min-w-0"><span className="block truncate text-[12px] font-medium">{company.company}</span><span className={`mt-1 block truncate text-[10px] ${muted}`}>{company.description || "No description supplied"}</span></span><span className={`self-center font-mono text-[10px] ${muted}`}>{company.ticker || "—"}</span><span className={`hidden self-center truncate text-[10px] sm:block ${muted}`}>{company.sector || "Unclassified"}</span><span className={`self-center text-right text-[10px] ${muted}`}>{count} →</span></button>;
                  }) : <EmptyState title="No companies found" message="Try a different search term." isLight={isLight} />}
                </div>
              ) : (
                <div>
                  {activeTab === "feed" && topPriorities.length > 0 && (
                    <section className="mx-3 mt-3 rounded-sm bg-[#F3F1EC] px-3 py-2.5 sm:mx-6">
                      <h2 className="mb-1.5 text-[13px] font-semibold leading-5 text-[#1A1A1A]">Needs attention</h2>
                      {topPriorities.map((record, index) => (
                        <button key={record.record_id} onClick={() => openRecord(record)} className={`flex w-full items-start gap-2 py-1.5 text-left ${index < topPriorities.length - 1 ? "mb-1.5" : ""}`}>
                          <span className="mt-[7px] h-2 w-2 shrink-0 rounded-full bg-[#C8372D]" />
                          <span className="min-w-0 flex-1">
                            <span style={{ fontFamily: '"Source Serif 4", Georgia, serif', fontWeight: 600, lineHeight: 1.35, color: "#1A1A1A" }} className="block truncate text-[17px]">{record.headline}</span>
                            <span className="block truncate text-[13px] leading-[1.35] text-[#6B6862]">{record.source_name || record.source_type || "Research"} · {formatDate(record.published_at, false)} · {record.company || record.ticker || "Company unavailable"}</span>
                          </span>
                        </button>
                      ))}
                    </section>
                  )}

                  {filteredRecords.length ? (
                    <div>
                      {filteredRecords.map((record, index) => {
                        const previous = filteredRecords[index - 1];
                        const day = record.published_at?.slice(0, 10);
                        const previousDay = previous?.published_at?.slice(0, 10);
                        const showDay = index === 0 || day !== previousDay;
                        return <div key={record.record_id}>
                          {showDay && <div className={`sticky top-[62px] z-10 border-b px-4 py-2 text-[11px] font-medium ${isLight ? "border-[#E7E5E0] bg-[#FAF9F6] text-[#55524D]" : "border-[#2a2e32] bg-[#111315] text-[#7f8790]"} sm:px-6`}>{formatDate(record.published_at)}</div>}
                          <ResearchRow refCallback={(node) => { rowRefs.current[record.record_id] = node; }} record={record} query={query} theme={theme} isSaved={savedIds.includes(record.record_id)} onOpen={() => openRecord(record)} onToggleSave={() => toggleSaved(record.record_id)} isLight={isLight} selected={selectedRecord?.record_id === record.record_id} />
                        </div>;
                      })}
                    </div>
                  ) : <EmptyState title={activeTab === "saved" ? "Nothing saved yet" : "No matching research"} message={activeTab === "saved" ? "Use the bookmark icon on any research row to save it for later." : activeTab === "watchlist" ? "No matching research for tracked companies. Add companies to your watchlist from a company profile." : "Try clearing a filter or searching for a different company, source, or keyword."} isLight={isLight} onClear={activeFilters.length ? clearFilters : undefined} />}
                </div>
              )}
            </div>

            <aside className={`${selectedRecord || selectedCompany ? "flex" : "hidden xl:flex"} fixed inset-0 z-30 flex-col overflow-y-auto border-l ${panel} xl:sticky xl:top-[62px] xl:h-[calc(100vh-62px)]`}>
              <div className={`flex h-[54px] shrink-0 items-center justify-between border-b px-5 ${border}`}>
                <div className={`text-[12px] font-medium ${muted}`}>{selectedRecord ? "Research detail" : selectedCompany ? "Company overview" : "Reading pane"}</div>
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <button onClick={() => setHelpOpen((value) => !value)} className={`flex h-7 w-7 items-center justify-center rounded-full border text-[12px] ${border} ${hover}`} aria-label="Keyboard shortcuts">?</button>
                    {helpOpen && <div className={`absolute right-0 top-9 z-40 w-52 rounded-md border p-3 shadow-lg ${panel} ${border}`}><div className="mb-2 text-[12px] font-semibold">Keyboard shortcuts</div><Shortcut keys="/" action="Focus search" muted={muted} /><Shortcut keys="j / k" action="Move through results" muted={muted} /><Shortcut keys="Enter" action="Open selected result" muted={muted} /><Shortcut keys="b" action="Bookmark result" muted={muted} /><Shortcut keys="Esc" action="Close details" muted={muted} /></div>}
                  </div>
                  {(selectedRecord || selectedCompany) && <button onClick={() => { setSelectedRecord(null); setSelectedCompany(null); }} className={`rounded-md px-2 py-1 text-lg ${muted} ${hover}`} aria-label="Close detail pane">×</button>}
                </div>
              </div>

              {selectedRecord ? (
                <div className="flex-1">
                  <div className="p-5">
                    <div className="mb-3 flex flex-wrap items-center gap-2"><span className={`rounded border px-2 py-1 text-[9px] ${border} ${muted}`}>{selectedRecord.source_type || "Unclassified"}</span><span className={`font-mono text-[9px] ${muted}`}>{selectedRecord.record_id}</span></div>
                    <h2 style={{ fontFamily: '"Source Serif 4", Georgia, serif', fontWeight: 600, lineHeight: 1.25 }} className="text-[22px] tracking-[-0.02em]">{selectedRecord.headline || "Untitled research record"}</h2>
                    <div className="mt-3 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[13px] text-[#6B6862]"><span>{selectedRecord.source_name || "Source unavailable"}</span><span>·</span><span>{formatDate(selectedRecord.published_at)}</span>{selectedRecord.company && <><span>·</span><span>{selectedRecord.company}</span></>}</div>
                    <div className="mt-3 flex items-center gap-2">
                      <button onClick={() => toggleSaved(selectedRecord.record_id)} className={`h-8 flex-1 rounded-md border px-3 text-[12px] ${border} ${hover}`}>{savedIds.includes(selectedRecord.record_id) ? "★ Saved for later" : "☆ Save for later"}</button>
                      {selectedRecord.ticker && companyMap.has(selectedRecord.ticker.toUpperCase()) && <button onClick={() => { const company = companyMap.get(selectedRecord.ticker.toUpperCase()); if (company) openCompany(company); }} className={`h-8 rounded-md border px-3 text-[12px] ${border} ${hover}`}>Company ↗</button>}
                    </div>
                  </div>

                  <div className={`border-y px-5 py-4 ${border}`}>
                    <SectionLabel muted={muted}>At a glance</SectionLabel>
                    <p className="mt-2 line-clamp-2 whitespace-pre-wrap text-[14px] leading-6 text-[#55524D]">{selectedRecord.content || "No summary content was supplied for this record."}</p>
                  </div>

                  <div className={`border-b px-5 py-4 ${border}`}>
                    <SectionLabel muted={muted}>Why this was flagged</SectionLabel>
                    <div className="mt-3 flex items-center gap-3"><PriorityDot score={priorityFor(selectedRecord)} /><div className="flex-1"><div className="text-[11px] font-medium">{priorityLabel(priorityFor(selectedRecord))} priority</div><div className={`mt-0.5 text-[10px] ${muted}`}>Heuristic score · {priorityFor(selectedRecord)}/100</div></div></div>
                    <p className={`mt-2 text-[10px] leading-5 ${muted}`}>{priorityReason(selectedRecord)}</p>
                  </div>

                  <div className={`border-b px-5 py-4 ${border}`}>
                    <SectionLabel muted={muted}>Metadata</SectionLabel>
                    <div className="mt-2 space-y-0.5">
                      <DetailMeta label="Company" value={selectedRecord.company} muted={muted} />
                      <DetailMeta label="Ticker" value={selectedRecord.ticker} muted={muted} />
                      <DetailMeta label="Sector" value={selectedRecord.sector} muted={muted} />
                      <DetailMeta label="Industry" value={selectedRecord.industry} muted={muted} />
                      <DetailMeta label="Country" value={selectedRecord.country} muted={muted} />
                      <DetailMeta label="Language" value={selectedRecord.language} muted={muted} />
                    </div>
                  </div>

                  <div className={`border-b px-5 py-4 ${border}`}>
                    <SectionLabel muted={muted}>Tags</SectionLabel>
                    <div className="mt-2 flex flex-wrap gap-1.5">{tagsOf(selectedRecord.tags).length ? tagsOf(selectedRecord.tags).map((tag) => <button key={tag} onClick={() => { setActiveTag(tag); setActiveTab("feed"); setSelectedRecord(null); }} className={`rounded-md border px-2 py-1 text-[10px] ${isLight ? "border-[#d9e0e6] bg-[#F2F1ED] text-[#55524D]" : "border-[#394047] bg-[#20262a] text-[#c5cbd0]"}`}>#{tag}</button>) : <span className={`text-[10px] ${muted}`}>No tags supplied</span>}</div>
                  </div>

                  <div className="p-5">
                    <SectionLabel muted={muted}>Original source document</SectionLabel>
                    {documentLoading ? <div className={`mt-3 text-[11px] ${muted}`}>Loading document…</div> : <pre className={`mt-3 max-h-[280px] overflow-auto whitespace-pre-wrap break-words rounded-md border p-3 font-mono text-[10px] leading-5 ${border} ${isLight ? "bg-[#FFFFFF] text-[#55524D]" : "bg-[#101214] text-[#b6bdc3]"}`}>{documentText || "No source document available."}</pre>}
                    <div className={`mt-2 break-all text-[10px] ${muted}`}>{selectedRecord.source_path || "No source path supplied"}</div>
                    {selectedRecord.source_path && <a href={`${DATA_ROOT}/${selectedRecord.source_path.split("/").map((part) => encodeURIComponent(part)).join("/")}`} target="_blank" rel="noreferrer" className="mt-4 flex w-full items-center justify-center gap-2 rounded-md bg-[#1F4FD8] px-4 py-3 text-[13px] font-semibold text-white transition-colors hover:bg-[#1943B8] focus:outline-none focus:ring-2 focus:ring-[#1F4FD8]/40">Open original source <span aria-hidden="true">↗</span></a>}
                  </div>
                </div>
              ) : selectedCompany ? (
                <div className="p-5">
                  <div className={`flex h-10 w-10 items-center justify-center rounded-lg border font-mono text-[11px] ${border}`}>{selectedCompany.ticker?.slice(0, 3) || "CO"}</div>
                  <h2 className="mt-4 text-[17px] font-semibold">{selectedCompany.company}</h2>
                  <p className={`mt-2 text-[11px] leading-5 ${muted}`}>{selectedCompany.description || "No company description supplied."}</p>
                  <div className={`mt-5 border-y py-3 ${border}`}><DetailMeta label="Ticker" value={selectedCompany.ticker} muted={muted} /><DetailMeta label="Sector" value={selectedCompany.sector} muted={muted} /><DetailMeta label="Industry" value={selectedCompany.industry} muted={muted} /><DetailMeta label="Geography" value={selectedCompany.geography} muted={muted} /><DetailMeta label="Aliases" value={selectedCompany.aliases} muted={muted} /></div>
                  <button onClick={() => toggleWatchlist(selectedCompany.ticker)} className={`mt-4 w-full rounded-md border px-3 py-2.5 text-[11px] ${border} ${hover}`}>{watchlistTickers.includes(selectedCompany.ticker) ? "✓ In watchlist — remove" : "+ Add to watchlist"}</button>
                  <div className={`mt-6 text-[10px] ${muted}`}>{companyRecords.length} related research records</div>
                  {companyRecords.slice(0, 8).map((record) => <button key={record.record_id} onClick={() => openRecord(record)} className={`block w-full border-b py-3 text-left ${border} ${hover}`}><span className="line-clamp-2 text-[11px] leading-4">{record.headline}</span><span className={`mt-1 block text-[9px] ${muted}`}>{formatDate(record.published_at, false)} · {record.source_type}</span></button>)}
                </div>
              ) : (
                <div className="flex-1" />
              )}
            </aside>
          </div>
        </section>
      </div>
    </main>
  );
}

function FilterSelect({ label, value, defaultValue, onChange, options, isLight }: { label: string; value: string; defaultValue: string; onChange: (value: string) => void; options: string[]; isLight: boolean }) {
  const selected = value !== defaultValue;
  return <div className={`relative inline-flex h-8 shrink-0 items-center gap-1 rounded-full border px-3 text-[13px] ${isLight ? "border-[#E7E5E0] bg-white text-[#55524D]" : "border-[#E7E5E0] bg-[#FFFFFF] text-[#55524D]"}`} style={{ borderRadius: 16 }}>
    <span>{selected ? `${label}: ${value}` : label}</span>
    {selected && <button type="button" aria-label={`Clear ${label} filter`} onClick={() => onChange(defaultValue)} className="relative z-10 ml-0.5 text-[13px] leading-none text-[#6B6862]">×</button>}
    <select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)} className="absolute inset-0 h-full w-full cursor-pointer opacity-0" style={{ maxWidth: "100%"}}>{options.map((option) => <option key={option} className="bg-white text-[#1A1A1A]">{option}</option>)}</select>
  </div>;
}

function PriorityDot({ score }: { score: number }) {
  const label = priorityLabel(score);
  const style = score >= 75
    ? "bg-[#FCE8E6] text-[#C8372D]"
    : score >= 58
      ? "bg-[#F8EEDC] text-[#8D5B0D]"
      : "bg-[#EFEEEA] text-[#66625C]";
  return <span className={`inline-flex items-center rounded-[4px] px-1.5 py-0.5 text-[10px] font-semibold ${style}`} title={`${score}/100 rule-based priority`}>{label}</span>;
}

function priorityReasonShort(record: ResearchRecord) {
  const text = `${record.headline} ${record.content} ${record.tags} ${record.source_type}`.toLowerCase();
  if (/acquisition|merger|stake|buyback/.test(text)) return "Corporate action identified";
  if (/regulatory|investigation|penalty|court|sebi|rbi|approval/.test(text)) return "Regulatory or legal development";
  if (/results|earnings|guidance|profit|revenue|margin|quarter/.test(text)) return "Earnings or financial update";
  if (/default|downgrade|upgrade|rating agency|credit/.test(text)) return "Credit or rating signal";
  return "Priority terms matched";
}

function priorityReason(record: ResearchRecord) {
  const text = `${record.headline} ${record.content} ${record.tags} ${record.source_type}`.toLowerCase();
  const reasons: string[] = [];
  if (/results|earnings|guidance|profit|revenue|margin|quarter/.test(text)) reasons.push("earnings or financial performance terms");
  if (/regulatory|order|approval|investigation|penalty|court|sebi|rbi/.test(text)) reasons.push("regulatory or legal terms");
  if (/acquisition|merger|stake|buyback|fundrais|default|downgrade|upgrade/.test(text)) reasons.push("corporate action or credit terms");
  if (/exchange filing|company announcement|rating agency/.test(text)) reasons.push("formal disclosure source type");
  return reasons.length ? `Raised because the record contains ${reasons.join(", ")}. This is a keyword-based triage explanation, not a model assessment.` : "No high-signal keyword pattern matched. The routine score is a navigation aid only, not a judgement about the investment significance of the story.";
}

function ResearchRow({ record, query, theme, isSaved, onOpen, onToggleSave, isLight, selected = false, refCallback }: { record: ResearchRecord; query: string; theme: Theme; isSaved: boolean; onOpen: () => void; onToggleSave: () => void; isLight: boolean; selected?: boolean; refCallback?: (node: HTMLButtonElement | null) => void }) {
  const muted = isLight ? "text-[#55524D]" : "text-[#B7B2A9]";
  const border = isLight ? "border-[#E7E5E0]" : "border-[#373530]";
  const hover = isLight ? "hover:bg-[#F2F1ED]" : "hover:bg-[#2A2926]";
  const score = priorityFor(record);
  const rowBackground = selected ? "bg-[#EEF3FF] border-l-[3px] border-l-[#1F4FD8] pl-[9px]" : "border-l-[3px] border-l-transparent";
  return <div className={`group flex gap-2 border-b px-3 py-3 transition-colors sm:gap-3 sm:px-6 ${border} ${hover} ${rowBackground}`}>
    <button ref={refCallback} onClick={onOpen} className="min-w-0 flex-1 text-left outline-none focus-visible:ring-1 focus-visible:ring-[#1F4FD8]">
      <div className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-1">
        <span className={`text-[13px] text-[#6B6862]`}>{record.source_name || record.source_type || "Research"}</span>
        <span className={`text-[13px] text-[#6B6862]`}>·</span>
        <span className={`text-[13px] text-[#6B6862]`}>{formatDate(record.published_at, false)}</span>
        {record.company && <><span className={`text-[13px] text-[#6B6862]`}>·</span><span className={`max-w-[180px] truncate text-[13px] text-[#6B6862]`}>{highlight(record.company, query, theme)}</span></>}
      </div>
      <h3 style={{ fontFamily: '"Source Serif 4", Georgia, serif', fontWeight: 600, lineHeight: 1.35, color: "#1A1A1A" }} className="mt-1.5 flex items-start gap-2 text-[17px] tracking-[-0.01em]">
        {score >= 75 && <span className="mt-[7px] h-2 w-2 shrink-0 rounded-full bg-[#C8372D]" />}
        {score >= 58 && score < 75 && <span className="mt-[7px] h-2 w-2 shrink-0 rounded-full bg-[#B7791F]" />}
        <span>{highlight(record.headline || "Untitled research record", query, theme)}</span>
      </h3>
      {record.content && <p className="mt-1 line-clamp-2 text-[14px] leading-5 text-[#55524D]">{highlight(record.content, query, theme)}</p>}
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        {tagsOf(record.tags).slice(0, 3).map((tag) => <span key={tag} className={`rounded-[4px] border px-1.5 py-0.5 text-[10px] ${isLight ? "border-[#E7E5E0] bg-[#F2F1ED] text-[#55524D]" : "border-[#30363b] bg-[#1a1e21] text-[#9ba4ac]"}`}>{tag}</span>)}
      </div>
    </button>
    <button onClick={onToggleSave} title={isSaved ? "Remove from Read later" : "Save to Read later"} aria-label={isSaved ? "Remove bookmark" : "Bookmark"} className={`mt-1 h-7 w-7 shrink-0 rounded-md text-[15px] transition-colors ${isSaved ? "text-[#B7791F]" : `${muted} opacity-0 group-hover:opacity-100 focus:opacity-100`} ${hover}`}>{isSaved ? "★" : "☆"}</button>
  </div>;
}

function EmptyState({ title, message, isLight, onClear }: { title: string; message: string; isLight: boolean; onClear?: () => void }) {
  return <div className="flex min-h-[250px] flex-col items-center justify-center px-6 py-12 text-center"><div className={`flex h-10 w-10 items-center justify-center rounded-xl border text-lg ${isLight ? "border-[#E7E5E0] bg-white text-[#7b848d]" : "border-[#30363b] bg-[#181b1e] text-[#858d95]"}`}>⌕</div><h3 className="mt-4 text-[12px] font-semibold">{title}</h3><p className={`mt-2 max-w-xs text-[11px] leading-5 ${isLight ? "text-[#55524D]" : "text-[#B7B2A9]"}`}>{message}</p>{onClear && <button onClick={onClear} className={`mt-4 rounded-md border px-3 py-2 text-[10px] ${isLight ? "border-[#E7E5E0]" : "border-[#383e44]"}`}>Clear filters</button>}</div>;
}

function CompanyFact({ label, value, isLight }: { label: string; value: string; isLight: boolean }) {
  return <div className={`border-b border-r px-4 py-3 ${isLight ? "border-[#E7E5E0]" : "border-[#2a2e32]"}`}><div className={`text-[9px] uppercase tracking-[0.08em] ${isLight ? "text-[#55524D]" : "text-[#7f8790]"}`}>{label}</div><div className="mt-1.5 break-words text-[11px]">{value || "Not supplied"}</div></div>;
}

function DetailMeta({ label, value, muted }: { label: string; value: string; muted: string }) {
  return <div className="flex items-start justify-between gap-4 py-2"><span className={`shrink-0 text-[13px] text-[#6B6862]`}>{label}</span><span className="max-w-[62%] break-words text-right text-[13px] text-[#1A1A1A]">{value || "Not supplied"}</span></div>;
}

function SectionLabel({ children, muted }: { children: React.ReactNode; muted: string }) {
  return <div className={`text-[12px] font-semibold tracking-normal ${muted}`}>{children}</div>;
}

function Shortcut({ keys, action, muted }: { keys: string; action: string; muted: string }) {
  return <div className="mt-3 flex items-center justify-between gap-3"><span className={`text-[10px] ${muted}`}>{action}</span><kbd className="rounded border border-current/20 px-1.5 py-0.5 font-mono text-[9px]">{keys}</kbd></div>;
}
