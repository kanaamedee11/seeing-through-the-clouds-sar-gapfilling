// Image tiles for the gap-filling paper figure (Bujumbura and the Imbo plain): Earth Engine export script.
// Paste into https://code.earthengine.google.com , press Run, then press "Run" on all 4 tasks in the Tasks tab.
var AOI = ee.Geometry.Rectangle([29.28, -3.52, 29.50, -3.28]);
var FOLDER = 'GapFillImages', CRS = 'EPSG:32735', SCALE = 100;
function stackBands(list) {
  return ee.Image(ee.List(list).iterate(function (cur, acc) { return ee.Image(acc).addBands(ee.Image(cur)); }, ee.Image([])));
}
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(AOI).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70));
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(AOI).filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
function masked(im) { var scl = im.select('SCL'); return im.updateMask(scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7))); }
function safe(col, band, reducer) {   // empty month -> fully masked image with the right band name
  var empty = ee.Image(0).rename(band).selfMask();
  var img = ee.Image(ee.Algorithms.If(col.size().gt(0), reducer === 'median' ? col.median() : col.mean(), empty));
  return img.select([band]);
}
function month(y, m) {
  var a = ee.Date.fromYMD(y, m, 1), b = a.advance(1, 'month');
  var tag = ee.String('_').cat(ee.Number(y).format('%d')).cat('_').cat(ee.Number(m).format('%02d'));
  var ndCol = s2.filterDate(a, b).map(function (im) { return masked(im).normalizedDifference(['B8', 'B4']).rename('n'); });
  var nd = safe(ndCol, 'n', 'median').multiply(1000).rename(ee.String('ndvi').cat(tag));
  var r = s1.filterDate(a, b);
  var vv = safe(r.select('VV'), 'VV', 'mean').multiply(100).rename(ee.String('vv').cat(tag));
  var vh = safe(r.select('VH'), 'VH', 'mean').multiply(100).rename(ee.String('vh').cat(tag));
  return ee.Image.cat([nd, vv, vh]);
}
var months = [];
for (var y = 2019; y <= 2025; y++) for (var m = 1; m <= 12; m++) months.push([y, m]);
var stack = stackBands(ee.List(months.map(function (p) { return month(p[0], p[1]); }))).unmask(-32768).toInt16().clip(AOI);
Export.image.toDrive({image: stack, description: 'gapfill_tile_monthly', folder: FOLDER, region: AOI, scale: SCALE, crs: CRS, maxPixels: 1e10});
var srtm = ee.Image('USGS/SRTMGL1_003');
var stat = ee.Image.cat([ee.Image('ESA/WorldCover/v200/2021').select('Map').rename('wc'), srtm.rename('elev_m'), ee.Terrain.slope(srtm).rename('slope_deg')]).toInt16().clip(AOI);
Export.image.toDrive({image: stat, description: 'gapfill_tile_static', folder: FOLDER, region: AOI, scale: SCALE, crs: CRS, maxPixels: 1e10});
// true-colour at 20 m for a clear month and a cloudy month of 2024, for display
function rgb(y, m) {
  var a = ee.Date.fromYMD(y, m, 1);
  var col = s2.filterDate(a, a.advance(1, 'month')).map(masked);
  var img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median().select(['B4', 'B3', 'B2']), ee.Image.constant([0, 0, 0]).rename(['B4', 'B3', 'B2']).selfMask()));
  return img.divide(3000).multiply(255).clamp(0, 255).toUint8();
}
Export.image.toDrive({image: ee.Image.cat([rgb(2024, 7).rename(['r7', 'g7', 'b7']), rgb(2024, 11).rename(['r11', 'g11', 'b11'])]).unmask(0).clip(AOI), description: 'gapfill_tile_truecolour', folder: FOLDER, region: AOI, scale: 20, crs: CRS, maxPixels: 1e10});
// monthly rainfall at the tile (CHIRPS is 5 km, so one value per month)
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');
var rain = ee.FeatureCollection(ee.List(months.map(function (p) { return p; })).map(function (p) {
  p = ee.List(p); var a = ee.Date.fromYMD(p.get(0), p.get(1), 1);
  return ee.Feature(null, {year: p.get(0), month: p.get(1), rain_mm: chirps.filterDate(a, a.advance(1, 'month')).sum().reduceRegion({reducer: ee.Reducer.mean(), geometry: AOI, scale: 5566}).get('precipitation')});
}));
Export.table.toDrive({collection: rain, description: 'gapfill_tile_rain', folder: FOLDER, fileFormat: 'CSV'});
