<p align="center">
  <img src="https://img.shields.io/badge/BRIN-Bharat_Road_Intelligence_Network-8C1C2B?style=for-the-badge&labelColor=1C2A3A" alt="BRIN Badge"/>
</p>

<h1 align="center">🛣️ BRIN — Bharat Road Intelligence Network</h1>

<p align="center">
  <em>Turn every AI-powered vehicle into a citizen road sensor.<br/>Detect hazards offline, broadcast over LoRa, auto-raise PWD repair tickets.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-3776AB?logo=python&logoColor=white" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/flask-3.0+-000000?logo=flask&logoColor=white" alt="Flask 3.0+"/>
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-00FFFF?logo=yolo&logoColor=white" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/LoRa-865_MHz_(India_ISM)-FF6600" alt="LoRa 865 MHz"/>
  <img src="https://img.shields.io/badge/license-MIT-1A9E6E" alt="MIT License"/>
</p>

---

## 📌 Overview

**BRIN** is an end-to-end IoT + AI pipeline that transforms AI-equipped vehicles into a **crowdsourced road damage detection network**. Vehicles running edge-AI (YOLOv8) detect road hazards — potholes, cracks, surface damage, faded lane markings — while driving, with **zero internet dependency**. Reports are broadcast over **LoRa radio** to roadside gateways, which relay them to a municipal cloud server. A live dashboard visualises damage as a heatmap, ranks repair priorities, and **automatically raises maintenance tickets** to the Public Works Department (PWD) once 3+ distinct vehicles confirm the same location.

### ✨ Key Features

| Feature | Description |
|:--------|:------------|
| **Edge AI Detection** | YOLOv8 road-damage model running on-device (Raspberry Pi + camera) — no cloud inference needed |
| **Fully Offline Vehicles** | Cars operate with zero internet; LoRa 865 MHz (India ISM band) handles all transmission |
| **LoRaWAN Gateway Bridge** | Roadside relay units with offline queuing and automatic retry on connectivity loss |
| **GPS Clustering** | Reports within ~11 m are automatically clustered into a single physical hazard location |
| **Multi-Vehicle Consensus** | PWD tickets fire only after **3+ distinct vehicles** confirm the same spot — eliminating false positives |
| **Live Municipal Dashboard** | Real-time Leaflet heatmap, repair priority ranking, and PWD ticket board with officer actions |
| **One-Command Demo** | Full simulation runs on a laptop with no hardware — see the entire pipeline in action instantly |

---

## 🏗️ System Architecture

```
 ┌─────────────────────┐          ┌───────────────────┐          ┌─────────────────────────┐
 │   AI CAR (Edge)     │  LoRa    │  ROADSIDE GATEWAY │  HTTP    │    CLOUD + DASHBOARD    │
 │                     │  865 MHz │                   │          │                         │
 │  📷 Camera          │ ───────► │  📡 LoRa Receiver │ ───────► │  🌐 Flask REST API      │
 │  🧠 YOLOv8 Detect   │ (no net) │  🔄 HTTP Uplink   │  (4G)   │  📊 GPS Clustering      │
 │  📍 GPS Geotag      │          │  💾 Offline Queue  │          │  🎫 PWD Auto-Ticketing  │
 │  ⚡ Risk Scoring    │          │                   │          │  🗺️  Leaflet Heatmap     │
 │  📻 LoRa TX         │          └───────────────────┘          │  📋 Priority Ranking    │
 └─────────────────────┘                                         └─────────────────────────┘
         ×N cars                        ×M gateways                    ×1 server
```

**Data flow:** `Camera Frame` → `YOLOv8 Inference` → `GPS Geotag` → `Risk Score` → `LoRa Broadcast` → `Gateway RX` → `HTTP POST` → `Cluster & Deduplicate` → `PWD Ticket (at 3+ confirmations)`

---

## 🚀 Quick Start

### One-Command Demo (No Hardware Required)

Everything runs in **simulation mode** by default. A fleet of 12 virtual AI Cars drives across Bengaluru hotspots, and you can watch the dashboard come alive.

```bash
git clone https://github.com/your-username/brin.git
cd brin
bash run_demo.sh
```

Then open **[http://localhost:5001](http://localhost:5001)** in your browser.

> **What you'll see:**
> - The heatmap filling in with hazard clusters
> - The repair priority list ranking locations by risk severity
> - PWD tickets auto-raised once 3+ distinct vehicles confirm the same spot

### Step-by-Step Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server + dashboard
cd server && python3 app.py
# → http://localhost:5001

# 3. Choose one of the simulation options below (in a new terminal):
```

#### Option A — Simulated Fleet (Recommended for Quick Demo)

Sends reports directly to the server, bypassing the LoRa layer. 12 virtual cars hit 5 Bengaluru hotspots.

```bash
cd simulator && python3 simulate_cars.py            # live stream
cd simulator && python3 simulate_cars.py --once      # seed 40 reports instantly
cd simulator && python3 simulate_cars.py --rate 0.5  # one report every 0.5s
```

#### Option B — Full Radio Path Simulation (Fake-LoRa over UDP)

Exercises the complete `edge → LoRa → gateway → server` pipeline on one laptop using UDP in place of LoRa radio.

```bash
# Terminal 2: Start a gateway
cd gateway && BRIN_SIM=1 python3 gateway_main.py --id GW-ORR-01

# Terminal 3+: Launch AI Cars (run multiple for multi-vehicle consensus)
cd edge && BRIN_SIM=1 python3 edge_main.py --car AICAR-07
cd edge && BRIN_SIM=1 python3 edge_main.py --car AICAR-03
cd edge && BRIN_SIM=1 python3 edge_main.py --car AICAR-12
```

> [!TIP]
> PWD tickets require **3 different car IDs** reporting the same spot. Use Option A (12 cars) or launch 3+ cars in Option B to see tickets fire.

---

## 🔧 Hardware Deployment

For real-world deployment on **Raspberry Pi** hardware, set `BRIN_SIM=0`. The same codebase automatically switches to real camera, GPS, and LoRa radio drivers.

### AI Car Edge Unit

**Hardware:** Raspberry Pi 4 + USB/CSI Camera + NEO-6M GPS + SX1278 LoRa Module

```bash
# Install hardware-specific dependencies
pip install ultralytics opencv-python numpy pynmea2 pyserial pyLoRa RPi.GPIO spidev

# Run the edge unit
BRIN_SIM=0 BRIN_MODEL=road_damage_yolov8.pt \
  python3 edge/edge_main.py --car AICAR-07
```

### Roadside Gateway

**Hardware:** Raspberry Pi + SX1301/SX127x LoRa Concentrator + 4G SIM Module

```bash
BRIN_SIM=0 BRIN_SERVER=https://brin.your-city.gov.in \
  python3 gateway/gateway_main.py --id GW-ORR-01
```

> [!NOTE]
> `BRIN_LORA_FREQ` defaults to **865.0 MHz** (India ISM band 865–867 MHz). Adjust for your region's regulations.

---

## 📁 Project Structure

```
brin/
├── protocol.py                  # Shared packet schema — LoRa (compact) + HTTP (JSON)
├── requirements.txt             # Python dependencies
├── run_demo.sh                  # One-command laptop demo launcher
│
├── edge/                        # 🚗 AI Car edge unit (runs on the vehicle)
│   ├── edge_main.py             #    Main loop: detect → geotag → score → LoRa TX
│   ├── detector.py              #    YOLOv8 road-damage detection (sim fallback)
│   ├── gps.py                   #    NEO-6M GPS reader (sim route fallback)
│   ├── risk.py                  #    Risk scoring engine (type + confidence + size)
│   └── lora_tx.py               #    LoRa transmitter — SX127x or UDP sim
│
├── gateway/                     # 📡 Roadside gateway (the "mailbox")
│   ├── gateway_main.py          #    Main loop: LoRa RX → HTTP POST, offline queue
│   └── lora_rx.py               #    LoRa receiver — SX127x or UDP sim
│
├── server/                      # ☁️ Municipal cloud server
│   ├── app.py                   #    Flask REST API + static dashboard host
│   ├── db.py                    #    SQLite store, GPS clustering, ticket table
│   └── pwd_pipeline.py          #    3-confirmation → PWD auto-ticket rule
│
├── dashboard/                   # 🗺️ Municipal Road-Health Dashboard (frontend)
│   ├── index.html               #    Dashboard layout
│   ├── app.js                   #    Real-time polling, heatmap, ticket board
│   └── style.css                #    UI styles
│
└── simulator/                   # 🎮 Laptop demo simulator
    └── simulate_cars.py         #    Synthetic fleet of 12 cars, Bengaluru hotspots
```

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/api/report` | Ingest a single hazard report (called by gateways) |
| `GET` | `/api/hazards` | Ranked hazard clusters for the heatmap & priority list |
| `GET` | `/api/tickets` | All PWD tickets with status |
| `POST` | `/api/tickets/<id>/progress` | Mark ticket as in-progress |
| `POST` | `/api/tickets/<id>/resolve` | Mark ticket as resolved |
| `POST` | `/api/tickets/<id>/reopen` | Reopen a resolved ticket |
| `GET` | `/api/stats` | Dashboard summary statistics |
| `POST` | `/api/reset` | Clear all reports and tickets (demo use) |

### Report Payload

```json
{
  "car_id":     "AICAR-07",
  "gateway_id": "GW-ORR-01",
  "hazard":     "pothole",
  "risk":       "high",
  "conf":       0.93,
  "lat":        12.9716,
  "lng":        77.5946
}
```

| Field | Type | Values |
|:------|:-----|:-------|
| `hazard` | `string` | `pothole` · `crack` · `damage` · `lane` |
| `risk` | `string` | `low` · `medium` · `high` · `critical` |
| `conf` | `float` | Model confidence (0.0 – 1.0) |
| `lat` / `lng` | `float` | GPS coordinates (WGS 84) |

### Response

```json
{
  "ok": true,
  "cluster_key": "12.9716_77.5946",
  "confirmations": 3,
  "threshold": 3,
  "ticket": "created"
}
```

The `ticket` field is `null` (below threshold), `"created"` (new ticket), or `"updated"` (existing ticket refreshed).

---

## ⚙️ Configuration

All configuration is done via **environment variables** — no config files needed.

| Variable | Scope | Default | Description |
|:---------|:------|:--------|:------------|
| `BRIN_SIM` | All | `1` | `1` = simulation mode (no hardware), `0` = real hardware |
| `BRIN_SERVER` | Gateway | `http://localhost:5001` | Municipal cloud server URL |
| `BRIN_MODEL` | Edge | `road_damage_yolov8.pt` | Path to YOLOv8 weights file |
| `BRIN_LORA_FREQ` | Edge / Gateway | `865.0` | LoRa frequency in MHz (India ISM: 865–867) |
| `BRIN_GPS_PORT` | Edge | `/dev/serial0` | Serial device for the GPS module |

### Server-Side Tuning

| Constant | File | Default | Description |
|:---------|:-----|:--------|:------------|
| `MIN_CONFIRMATIONS` | `server/db.py` | `3` | Distinct vehicles needed to auto-raise a PWD ticket |
| `CLUSTER_PRECISION` | `server/db.py` | `4` | GPS decimal places for clustering (~11 m at 4 decimals) |

---

## 📋 Protocol Specification

BRIN uses a **dual wire format** defined in [`protocol.py`](protocol.py):

| Format | Transport | Example |
|:-------|:----------|:--------|
| **LoRa** (pipe-delimited) | Car → Gateway | `1\|AICAR-07\|0\|2\|0.930\|12.971600\|77.594600\|1717000000` |
| **JSON** | Gateway → Server | Standard JSON body (see API Reference) |

The LoRa format uses compact integer codes for hazard types and risk levels to fit within LoRa's ~50–200 byte frame limit. Both formats are defined in a **single shared file** so edge, gateway, and server can never disagree on the packet schema.

---

## 🏭 Production Considerations

| Area | Prototype | Production Recommendation |
|:-----|:----------|:--------------------------|
| **Database** | SQLite (`brin.db`) | PostgreSQL + PostGIS (`cluster_key` → geohash / `ST_SnapToGrid`) |
| **PWD Integration** | Console log | Replace `dispatch_to_pwd()` in `pwd_pipeline.py` with HTTP call to your state's civic-grievance API |
| **Deployment** | `python app.py` | WSGI server (Gunicorn) behind Nginx, with systemd services |
| **Auth** | None | API key / mTLS for gateway → server communication |
| **Monitoring** | — | Prometheus + Grafana for gateway health, report volume, ticket SLAs |
| **Scaling** | Single server | Horizontal scaling with Redis pub/sub for real-time dashboard updates |

---

## 🗺️ Roadmap

- [ ] Real-time WebSocket push (replace 3s polling)
- [ ] Historical analytics & trend detection
- [ ] Multi-city support with configurable regions
- [ ] Mobile companion app for PWD field officers
- [ ] Integration with national road authority APIs
- [ ] MQTT broker support alongside LoRa
- [ ] Edge model retraining pipeline with field-collected images

---

## 🤝 Contributing

Contributions are welcome! Whether it's improving the detection model, adding new hazard types, or integrating with your city's PWD portal — we'd love your help.

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built with ❤️ for India's roads</strong><br/>
  <sub>Every pothole reported is a step towards safer streets.</sub>
</p>
