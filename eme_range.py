#!/usr/bin/env python3
"""
EME (moonbounce) range visualizer.

Two stations can work each other via the Moon when the Moon is above the
minimum elevation at BOTH places at the same time ("mutual moon window").

Modes
  * Coverage heatmap : % of time over the next N days a location shares a window with you
  * Reachable now    : where the Moon is up right now (only if it is also up at your station)
Click any map point for its upcoming windows and a home-vs-target elevation comparison.

Local:   pip install -r requirements.txt ; python eme_range.py   (optionally --lat/--lon)
Hosted:  gunicorn eme_range:server   (configure with env vars HOME_GRID, MIN_EL, DAYS, GRID)
Needs de421.bsp (17 MB) next to this file; it is downloaded automatically on first run.
"""
import argparse
import functools
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from dash.exceptions import PreventUpdate
from skyfield.api import Loader
from skyfield.framelib import itrs

def maiden_center(loc):
    """Center lat/lon of a 6-character Maidenhead locator."""
    loc = loc.strip()
    lon = (ord(loc[0].upper()) - 65) * 20 + int(loc[2]) * 2 + (ord(loc[4].lower()) - 97) / 12 + 1 / 24
    lat = (ord(loc[1].upper()) - 65) * 10 + int(loc[3]) + (ord(loc[5].lower()) - 97) / 24 + 1 / 48
    return lat - 90, lon - 180


# Defaults come from environment variables (set these on the host); CLI flags override locally.
# HOME_GRID is the club grid square; HOME_LAT / HOME_LON override it if given.
_glat, _glon = maiden_center(os.environ.get("HOME_GRID", "FM05ps"))
ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, default=float(os.environ.get("HOME_LAT", _glat)))
ap.add_argument("--lon", type=float, default=float(os.environ.get("HOME_LON", _glon)))
ap.add_argument("--min-el", type=float, default=float(os.environ.get("MIN_EL", 0)))
ap.add_argument("--days", type=int, default=int(os.environ.get("DAYS", 28)))
ap.add_argument("--grid", type=float, default=float(os.environ.get("GRID", 4)))
ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8050)))
# under gunicorn sys.argv holds gunicorn's flags, so only parse the CLI when run directly
args, _ = ap.parse_known_args(sys.argv[1:] if __name__ == "__main__" else [])

R_EARTH = 6371.0   # km, spherical-earth approximation
STEP_MIN = 10      # time resolution of the forecast
N_STEPS = args.days * 1440 // STEP_MIN + 6   # +1 h so the cached track can start on the hour

# keep de421.bsp next to this script so a deployed copy never needs to download it
load = Loader(os.path.dirname(os.path.abspath(__file__)))
ts = load.timescale()
eph = load("de421.bsp")


# ---------------------------------------------------------------- geometry ---
def unit_vec(lat, lon):
    """Geodetic lat/lon (deg) -> unit vector in the Earth-fixed (ITRS) frame."""
    phi = np.arctan((1 - 0.00669438) * np.tan(np.radians(lat)))  # geodetic -> geocentric
    lam = np.radians(lon)
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)


def moon_el(moon, U):
    """Topocentric moon elevation (deg), incl. parallax. moon:(T,3) km ITRS, U:(G,3) -> (T,G)."""
    D = moon @ U.T
    m2 = np.einsum("ij,ij->i", moon, moon)[:, None]
    rng = np.sqrt(m2 - 2 * R_EARTH * D + R_EARTH**2)
    return np.degrees(np.arcsin((D - R_EARTH) / rng))


def moon_xyz(t):
    return np.atleast_2d(eph["earth"].at(t).observe(eph["moon"]).frame_xyz(itrs).km.T)


def cur_bucket():
    return int(time.time() // 3600) * 6   # in units of 10 minutes, rounded to the hour


@functools.lru_cache(maxsize=2)
def moon_track(bucket):
    t0 = datetime.fromtimestamp(bucket * 600, timezone.utc)
    times = [t0 + timedelta(minutes=STEP_MIN * i) for i in range(N_STEPS)]
    return times, moon_xyz(ts.from_datetimes(times))


def locator(lat, lon):
    """6-character Maidenhead grid square."""
    lon, lat = min(lon + 180, 359.999), min(lat + 90, 179.999)
    return (chr(65 + int(lon // 20)) + chr(65 + int(lat // 10)) + str(int(lon % 20 // 2))
            + str(int(lat % 10)) + chr(97 + int(lon % 2 * 12)) + chr(97 + int(lat % 1 * 24)))


# ------------------------------------------------------------------- grid ----
LATS = np.arange(-90 + args.grid / 2, 90, args.grid)
LONS = np.arange(-180 + args.grid / 2, 180, args.grid)
GLAT, GLON = (a.ravel() for a in np.meshgrid(LATS, LONS, indexing="ij"))
GU = unit_vec(GLAT, GLON)
GLOC = [locator(a, b) for a, b in zip(GLAT, GLON)]


@functools.lru_cache(maxsize=4)
def coverage(bucket, lat, lon, min_el):
    """% of forecast time each grid cell shares a moon window with home; also home moon-up %."""
    _, moon = moon_track(bucket)
    home_up = moon_el(moon, unit_vec(lat, lon)[None, :])[:, 0] >= min_el
    count = np.zeros(len(GU))
    for i in range(0, len(moon), 256):
        el = moon_el(moon[i:i + 256], GU)
        count += ((el >= min_el) & home_up[i:i + 256, None]).sum(axis=0)
    return count / len(moon) * 100, home_up.mean() * 100


# ---------------------------------------------------------------- figures ----
def base_figure(title):
    fig = go.Figure()
    fig.update_geos(projection_type="equirectangular", showcountries=True, showcoastlines=True,
                    showland=True, landcolor="#f2f2f2", oceancolor="#dfeaf5", showocean=True,
                    lataxis_range=[-90, 90], lonaxis_range=[-180, 180])
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), uirevision="keep",
                      legend=dict(orientation="h", y=0.02), clickmode="event")
    return fig


def add_home(fig, lat, lon):
    fig.add_trace(go.Scattergeo(lat=[lat], lon=[lon], mode="markers", name="Home",
                                marker=dict(symbol="star", size=15, color="red",
                                            line=dict(width=1, color="black")),
                                hovertext=f"Home {locator(lat, lon)}", hoverinfo="text"))


def heat_figure(lat, lon, min_el):
    pct, home_pct = coverage(cur_bucket(), round(lat, 4), round(lon, 4), min_el)
    fig = base_figure(f"Share of next {args.days} days each location can be worked "
                      f"(home sees moon {home_pct:.0f}% of the time, min el {min_el:g}°)")
    fig.add_trace(go.Scattergeo(
        lat=GLAT, lon=GLON, mode="markers", name="Coverage",
        marker=dict(symbol="square", size=args.grid * 2.8, opacity=0.8, line_width=0, color=pct,
                    colorscale="Viridis", cmin=0, cmax=max(pct.max(), 1),
                    colorbar=dict(title="% of time", thickness=12)),
        hovertext=[f"{g}  {p:.0f}% of time" for g, p in zip(GLOC, pct)], hoverinfo="text"))
    add_home(fig, lat, lon)
    return fig


def visibility_curve(moon, min_el):
    """Lat/lon of the circle where the Moon sits exactly at min_el (the edge of its footprint)."""
    d = np.linalg.norm(moon)
    z = np.radians(90 - min_el)
    gamma = z - np.arcsin(R_EARTH * np.sin(z) / d)      # angular radius around the sub-lunar point
    phi, lam = np.arcsin(moon[2] / d), np.arctan2(moon[1], moon[0])
    th = np.radians(np.linspace(0, 360, 361))
    lat2 = np.arcsin(np.sin(phi) * np.cos(gamma) + np.cos(phi) * np.sin(gamma) * np.cos(th))
    lon2 = lam + np.arctan2(np.sin(th) * np.sin(gamma) * np.cos(phi),
                            np.cos(gamma) - np.sin(phi) * np.sin(lat2))
    lat = np.degrees(np.arctan(np.tan(lat2) / (1 - 0.00669438)))
    lon = (np.degrees(lon2) + 180) % 360 - 180
    brk = np.flatnonzero(np.abs(np.diff(lon)) > 180) + 1  # break the line at the dateline
    return np.insert(lat, brk, np.nan), np.insert(lon, brk, np.nan), phi, lam


def now_figure(lat, lon, min_el):
    moon = moon_xyz(ts.now())
    home_el = moon_el(moon, unit_vec(lat, lon)[None, :])[0, 0]
    el = moon_el(moon, GU)[0]
    c_lat, c_lon, phi, lam = visibility_curve(moon[0], min_el)
    status = ("anywhere inside the curve can work you" if home_el >= min_el else
              "moon is below the minimum elevation at home, so no one is reachable right now")
    fig = base_figure(f"Moon visibility now ({datetime.now(timezone.utc):%H:%M}Z), "
                      f"home moon el {home_el:+.1f}°: {status}")
    # near-invisible grid so hover/click still works for the details table
    fig.add_trace(go.Scattergeo(
        lat=GLAT, lon=GLON, mode="markers", showlegend=False,
        marker=dict(symbol="square", size=args.grid * 2.8, color="white", opacity=0.01, line_width=0),
        hovertext=[f"{g}  moon el {e:+.1f}°" for g, e in zip(GLOC, el)], hoverinfo="text"))
    fig.add_trace(go.Scattergeo(lat=c_lat, lon=c_lon, mode="lines", name=f"Moon at {min_el:g}° elevation",
                                line=dict(width=3, color="darkorange"), hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(
        lat=[np.degrees(np.arctan(np.tan(phi) / (1 - 0.00669438)))], lon=[np.degrees(lam)],
        mode="markers", name="Sub-lunar point",
        marker=dict(symbol="circle", size=13, color="gold", line=dict(width=2, color="black")),
        hovertext="Moon at zenith here", hoverinfo="text"))
    add_home(fig, lat, lon)
    return fig


# -------------------------------------------------------------------- app ----
app = Dash(__name__)
server = app.server   # WSGI entry point for gunicorn: gunicorn eme_range:server
app.layout = html.Div(style={"fontFamily": "sans-serif", "margin": "8px"}, children=[
    html.Div(style={"display": "flex", "gap": "24px", "alignItems": "center", "flexWrap": "wrap"}, children=[
        dcc.RadioItems(id="mode", value="heat", inline=True, options=[
            {"label": " Coverage heatmap", "value": "heat"},
            {"label": " Reachable right now", "value": "now"}]),
        html.Span(["Home lat ", dcc.Input(id="lat", type="number", value=args.lat, debounce=True, style={"width": 80})]),
        html.Span(["lon ", dcc.Input(id="lon", type="number", value=args.lon, debounce=True, style={"width": 80})]),
        html.Div(style={"width": 260}, children=[
            "Min moon elevation (°)",
            dcc.Slider(id="minel", min=0, max=30, step=1, value=args.min_el,
                       marks={0: "0", 10: "10", 20: "20", 30: "30"})]),
    ]),
    dcc.Graph(id="map", style={"height": "62vh"}),
    html.Div(id="details", style={"padding": "4px 8px"}),
    dcc.Interval(id="tick", interval=60_000),
])


@app.callback(Output("map", "figure"), Input("mode", "value"), Input("lat", "value"),
              Input("lon", "value"), Input("minel", "value"), Input("tick", "n_intervals"))
def update_map(mode, lat, lon, min_el, _):
    if lat is None or lon is None:
        raise PreventUpdate
    return heat_figure(lat, lon, min_el) if mode == "heat" else now_figure(lat, lon, min_el)


def fmt(dt):
    return dt.strftime("%a %d %b %H:%MZ")


@app.callback(Output("details", "children"), Input("map", "clickData"), Input("lat", "value"),
              Input("lon", "value"), Input("minel", "value"), Input("tick", "n_intervals"))
def show_details(click, lat, lon, min_el, _):
    if lat is None or lon is None:
        raise PreventUpdate
    if not click:
        return html.P("Click a point on the map to see its upcoming moon windows with your station.")
    p = click["points"][0]
    tlat, tlon = p["lat"], p["lon"]
    times, moon = moon_track(cur_bucket())
    h = moon_el(moon, unit_vec(lat, lon)[None, :])[:, 0]
    r = moon_el(moon, unit_vec(tlat, tlon)[None, :])[:, 0]
    common = (h >= min_el) & (r >= min_el)
    now = datetime.now(timezone.utc)
    i_now = max(0, int((now - times[0]).total_seconds() // (STEP_MIN * 60)))

    edges = np.diff(np.r_[False, common, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    rows = []
    for s, e in zip(starts, ends):
        end_t = times[e - 1] + timedelta(minutes=STEP_MIN)
        if end_t < now:
            continue
        b = s + int(np.minimum(h[s:e], r[s:e]).argmax())   # best = highest common elevation
        note = "in progress" if times[s] <= now else ""
        if e == len(times):
            note = (note + ", " if note else "") + "continues past forecast"
        rows.append([fmt(max(times[s], now) if times[s] <= now else times[s]), fmt(end_t),
                     f"{(e - s) * STEP_MIN // 60}h {(e - s) * STEP_MIN % 60:02d}m", fmt(times[b]),
                     f"{h[b]:.1f}°", f"{r[b]:.1f}°", "Home" if h[b] < r[b] else "Target", note])
        if len(rows) >= 10:
            break

    head = ["Window start", "Window end", "Length", "Best moment", "Home el", "Target el",
            "Lower end", "Note"]
    cell = {"border": "1px solid #ccc", "padding": "3px 8px", "textAlign": "center"}
    table = html.Table(style={"borderCollapse": "collapse", "fontSize": "14px"}, children=[
        html.Thead(html.Tr([html.Th(c, style=cell) for c in head])),
        html.Tbody([html.Tr([html.Td(c, style=cell) for c in row]) for row in rows])]) if rows else \
        html.P(f"No shared moon windows in the next {args.days} days at this elevation limit.")
    return html.Div([
        html.H4(f"Target {tlat:.1f}°, {tlon:.1f}° ({locator(tlat, tlon)}): "
                f"{common.mean() * 100:.0f}% of the next {args.days} days workable"),
        html.P(f"Right now the moon is at {h[i_now]:+.1f}° at home and {r[i_now]:+.1f}° at the target."),
        table,
        html.P("'Best moment' is when the lower of the two elevations peaks; 'Lower end' is the "
               "station that is closer to the horizon then (usually your limiting factor).",
               style={"color": "#666", "fontSize": "12px"}),
    ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=args.port)
