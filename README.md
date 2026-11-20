# Academic Digest

Her sabah o günün **okunmaya değer** akademik makalelerini mail atan, aynı veriyi web arayüzü,
JSON API ve Atom feed olarak da sunan otomasyon. arXiv'in günlük ~700 makalelik akışı olduğu gibi
gelmez: topluluk ilgisi (Hugging Face Papers, Hacker News), yayın yeri (NeurIPS/ICLR kabulü,
Nature), kod paylaşımı ve senin ilgi alanların puanlanır; geri çekilen makaleler, düzeltme
notları ve haber yazıları elenir.

[tech-digest](../tech-digest) ile aynı mimari: kaynaklar → SQLite → cluster → classify → score →
select → summarize → render → mail. Yenilikler: **enricher** aşaması, **web** paketi ve
**LLM altyapısı**.

## Nasıl çalışır

```
collect (3 saatte bir)  →  enrich  →  SQLite  →  digest (08:30 İstanbul)
                                        │          cluster → classify → score → select → summarize → render → mail
                                        └──────→  serve (web UI + /api + /feed.xml, aynı DB'yi okur)
```

- **Kaynaklar** (`config/sources.yaml`): arXiv günlük duyuruları (tek RSS'te birden çok kategori,
  yalnızca `new` + `cross`; `replace` eski makalenin yeni sürümü olduğu için atlanır),
  Hugging Face Daily Papers (upvote), Hacker News'te arxiv.org / nature.com / openreview.net…
  linkleri, Nature ve Nature Machine Intelligence (yalnızca araştırma makaleleri).
- **Enricher'lar:** collect'ten sonra pencere içindeki makalelere ek bilgi ekler.
  `arxiv_meta` arXiv API'den yazar yorumlarını ("Accepted at ICLR 2027", kod linki) ve birincil
  kategoriyi çeker; HN'de paylaşılan eski bir arXiv makalesinin özetini/yazarlarını da doldurur.
  `semantic_scholar` (varsayılan kapalı) atıf sayısı ve en yüksek yazar h-index'ini ekler.
- **Cluster'lama:** aynı makale arXiv id'si (abs/pdf/html, sürüm fark etmez), DOI, URL veya
  (arXiv id'si olmayan dergi sürümü için) başlık benzerliğiyle tek kayda birleşir.
- **Sınıflandırma:** önce arXiv kategorisi (birincil kategori +4), sonra başlık/özet anahtar
  kelimeleri. Bölümler: AI & ML, Language & LLMs, Vision & Multimodal, Security & Privacy,
  Systems & Software, Science.
- **Puanlama** (`config/scoring.yaml`): kaynak kademesi (arXiv tek başına 0 puan — her şeyi
  listeler, hiçbir şeyi önermez) + bağımsız kaynak kapsamı (HF + HN…) + etkileşim (HF upvote,
  HN puan/yorum) + yayın yeri (top venue, oral/spotlight) + kod + **ilgi alanların** + **takip
  ettiğin yazarlar** − cezalar (withdrawn, correction/retraction, haber/editoryal), sonra tazeliğe
  göre azaltma. Her puanın gerekçesi kayıtlı.
- **Seçim:** "Today's highlights" (ilk 5), sonra kategori başına en fazla 5 makale. Toplam
  sınır aşılırsa en kalabalık bölüm kırpılır, sessiz bölüm silinmez. Eşiği geçemeyen gün kısa
  mail gelir (`send_empty: false` ile hiç gelmez). Gönderilmiş makale bir daha gelmez — HN'de
  ertesi gün tekrar gündeme gelse bile.

## Komutlar

```bash
academicdigest collect [-s arXiv] [--no-enrich]   # kaynakları çek + enrich
academicdigest preview [--collect]                # göndermeden HTML üret + puan dökümü (ayar için)
academicdigest digest [--force]                   # bugünün digest'ini gönder (gün başına bir kez)
academicdigest send-test                          # SMTP ayarını dene
academicdigest serve [--port 8080]                # web arayüzü + API + feed
academicdigest llm-check ["prompt"]               # LLM sağlayıcısını dene (sağlayıcı eklenince)
academicdigest llm-usage                          # LLM token kullanımı
```

## Web arayüzü ve API

`academicdigest serve` CLI ile aynı veritabanını salt okunur kullanır.

| Sayfa | Ne |
|---|---|
| `/` | son digest'e yönlendirir |
| `/digests`, `/digests/{tarih}` | arşiv ve günlük digest; `?all=1` tüm adaylar + puan dökümü |
| `/digests/{tarih}/email` | gönderilen mailin birebir HTML'i |
| `/papers?q=…` | toplanan **tüm** makalelerde arama (başlık, özet, yazar, arXiv id) |
| `/papers/{id}` | makale detayı: hangi kaynaklarda görüldü, hangi digest'lere girdi |
| `/status` | kaynak başına son 24 saat / 7 gün, son digest durumları |
| `/feed.xml` | gönderilen digest'lerin Atom feed'i (RSS okuyucu için) |
| `/api/docs` | JSON API (OpenAPI): `/api/digests`, `/api/digests/latest`, `/api/digests/{tarih}?all=true`, `/api/papers`, `/api/papers/{id}`, `/api/status` |
| `/healthz` | izleme (auth gerektirmez) |

`.env`: `AD_WEB_HOST`/`AD_WEB_PORT` (varsayılan `127.0.0.1:8080`), `AD_WEB_BASE_URL` (maile
"view on the web" linki ve feed linkleri), `AD_WEB_USER` + `AD_WEB_PASSWORD` (ikisi de doluysa
HTTP basic auth). Önüne bir reverse proxy koy, örneğin Caddy:

```
papers.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

## LLM altyapısı

Henüz hiçbir sağlayıcı bağlı değil (`AD_LLM_PROVIDER=none`); her şey kural tabanlı çalışır.
Hazır olanlar (`src/academicdigest/llm/`):

- **`LLMClient` protokolü** — pipeline hiçbir SDK'yı doğrudan çağırmaz, `complete(LLMRequest) -> LLMResponse` ister.
- **Sağlayıcı kaydı** — `@register_provider("ad")`, `AD_LLM_PROVIDER` ile seçilir.
- **Önbellek + kullanım kaydı** — `llm_calls` tablosu; aynı model + istek ikinci kez para harcamaz, token sayıları tutulur (`llm-usage`).
- **Bütçe** — `AD_LLM_MAX_CALLS`: bir komut çalışmasında en fazla bu kadar ücretli çağrı (cache isabetleri sayılmaz). Aşılınca kalan makaleler kural puanı / özet alıntısıyla devam eder.
- **Prompt şablonları** — `llm/prompts/rate.j2` (makaleyi 0-10 puanla, JSON) ve `summarize.j2`; kod değiştirmeden düzenlenebilir. `parse_json` kod bloğu / gevezelik içindeki JSON'u ayıklar.
- **Pipeline bağlantıları** — `engine: hybrid`: kural puanından sonra ilk `rerank_top_k` makaleyi LLM, `llm.profile`'daki araştırma profiline göre puanlar; skor `(puan − 5) × weight` kadar oynar, gerekçe web'de görünür. `summarizer: llm`: özet alıntısı yerine LLM özeti (`llm.language` ile Türkçe olabilir). Hata olursa o makale kural puanı / alıntıyla kalır.

Sağlayıcı eklemek tek dosya, örneğin `src/academicdigest/llm/providers.py`:

```python
import httpx

from .base import LLMError, LLMRequest, LLMResponse, register_provider


@register_provider("myprovider")
class MyProvider:
    name = "myprovider"

    def __init__(self, settings):
        self.model = settings.llm_model
        self.http = httpx.Client(base_url=settings.llm_base_url, timeout=settings.llm_timeout,
                                 headers={"Authorization": f"Bearer {settings.llm_api_key}"})

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            ...  # API çağrısı: request.system, request.prompt, request.max_tokens, request.json
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        return LLMResponse(text=..., model=self.model, input_tokens=..., output_tokens=...)
```

Sonra `llm/__init__.py` içinde `from . import providers` ile import et, `.env`'de
`AD_LLM_PROVIDER=myprovider` + model/anahtar ver, `academicdigest llm-check` ile dene,
`scoring.yaml`'da `engine: hybrid` / `summarizer: llm` yap.

## Lokal geliştirme

```bash
uv sync
cp .env.example .env        # SMTP bilgilerini gir
uv run academicdigest collect
uv run academicdigest preview   # preview.html'i tarayıcıda aç
uv run academicdigest serve     # http://127.0.0.1:8080
uv run pytest
```

## VPS kurulumu

Gereken: systemd'li bir Linux, `git`, [`uv`](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh   # uv yoksa
git clone <repo-url> academic-digest && cd academic-digest
sudo ./deploy/install.sh                                    # /opt/academic-digest + timer'lar + web servisi
sudo -u academicdigest nano /opt/academic-digest/.env       # SMTP, AD_WEB_*
cd /opt/academic-digest && sudo -u academicdigest .venv/bin/academicdigest send-test
sudo -u academicdigest .venv/bin/academicdigest collect     # ilk doldurma
sudo systemctl restart academicdigest-web                   # .env değiştiyse
```

Güncelleme: repoda `git pull`, ardından tekrar `sudo ./deploy/install.sh` (`.env` ve `data/` korunur,
web servisi yeniden başlar).

İzleme:

```bash
systemctl list-timers 'academicdigest-*'
journalctl -u academicdigest-digest -n 50
journalctl -u academicdigest-collect --since today
journalctl -u academicdigest-web -f
curl -s 127.0.0.1:8080/healthz
```

**Zamanlama:** arXiv duyuruları Pazar–Perşembe akşamı (ABD Doğu saati) çıkar, RSS gece yarısı
(ABD Doğu) güncellenir; digest bu yüzden 08:30 İstanbul'da gider. Cumartesi/Pazar sabahları
arXiv'den yeni makale gelmez — o günler sadece HF/HN/dergi kaynaklı makaleler çıkar
(`send_empty: false` ile boş günlerde mail gelmez).

### Gmail SMTP

1. Google hesabında 2 adımlı doğrulamayı aç.
2. <https://myaccount.google.com/apppasswords> üzerinden bir App Password oluştur.
3. `.env`: `AD_SMTP_HOST=smtp.gmail.com`, `AD_SMTP_PORT=587`, `AD_SMTP_SECURITY=starttls`,
   `AD_SMTP_USER=adresin@gmail.com`, `AD_SMTP_PASSWORD=<app password>`.

### Kendi mail sunucuna geçiş

Kod değişmez, sadece `.env` değişir (`AD_SMTP_HOST`, port, `starttls`/`ssl`, kullanıcı/şifre).
Spam'e düşmemek için sunucuda SPF, DKIM, DMARC ve reverse DNS (PTR) kayıtlarını ayarla.

## Ayar (tuning)

İlk 1-2 hafta `academicdigest preview` çıktısına veya web'de bir digest'in "all candidates +
scores" görünümüne bakıp ayarla — kod değiştirmek gerekmez:

- `config/scoring.yaml` → **`interests`** (asıl kişiselleştirme düğmesi: kendi konularının
  regex'leri ve ağırlıkları), **`watch_authors`**, `boost_patterns` / `penalty_patterns`,
  kategoriler, eşikler.
- `config/sources.yaml` → arXiv `categories` listesi, HF `min_upvotes`, HN `min_points`,
  dergiler (`type: journal` herhangi bir RSS/Atom ile çalışır), kaynak ağırlıkları (`weight`).

## Genişleme noktaları

| Ne | Nerede | Nasıl |
|---|---|---|
| Yeni kaynak | `src/academicdigest/sources/` | `Source` alt sınıfı + `@register("tip")` + YAML satırı |
| Yeni enricher | `src/academicdigest/enrich/` | `Enricher` alt sınıfı (`wants`, `enrich` → `ItemPatch`) + `@register` + `enrichers:` satırı |
| LLM sağlayıcı | `src/academicdigest/llm/` | `@register_provider("ad")`, `complete(LLMRequest)`; bkz. yukarı |
| LLM puanlama / özet | `pipeline/score.py`, `pipeline/summarize.py` | hazır: `engine: hybrid`, `summarizer: llm`; prompt'lar `llm/prompts/*.j2` |
| Başka teslimat kanalı | `delivery/` | `Notifier` protokolü (`send(Rendered)`) |
| Web / API | `web/` | `queries.py` (dict döndüren sorgular) + `app.py` (route'lar) + `templates/` |
| Postgres | `.env` | `AD_DATABASE_URL=postgresql+psycopg://…` (+ `psycopg` bağımlılığı) |
