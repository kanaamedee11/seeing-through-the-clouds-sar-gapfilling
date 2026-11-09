// Burundi SAR-optical gap-filling study: Earth Engine export script (Code Editor).
// Paste into https://code.earthengine.google.com , press Run, then press "Run" on all 8 tasks in the Tasks tab.
// No API keys needed. Wait until all 8 tasks show as completed, then download the Drive folder 'BurundiGapFill'.

// ---------------- settings ----------------
var REGION = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_na', 'Burundi')).geometry();
var Y0 = 2019, Y1 = 2025;
var FOLDER = 'BurundiGapFill';
var SCALE = 20;
var CLASSES = [10, 20, 30, 40, 50, 60, 90];     // WorldCover: tree, shrub, grass, crop, built, bare, wetland
var PER_CLASS = 1100;                           // sample points per class: 7,700 points in all

// ---------------- static layers and the sample ----------------
var wc = ee.Image('ESA/WorldCover/v200/2021').select('Map').rename('wc');
var srtm = ee.Image('USGS/SRTMGL1_003');
var pts = wc.stratifiedSample({
  numPoints: 0, classBand: 'wc', region: REGION, scale: SCALE, seed: 7,
  classValues: CLASSES, classPoints: CLASSES.map(function () { return PER_CLASS; }),
  geometries: true, dropNulls: true, tileScale: 8
}).map(function (f) {
  var c = f.geometry().coordinates();
  return f.set({lon: ee.Number(c.get(0)).multiply(1e5).round().divide(1e5), lat: ee.Number(c.get(1)).multiply(1e5).round().divide(1e5)});
});
var statics = ee.Image.cat([srtm.rename('elev_m'), ee.Terrain.slope(srtm).rename('slope_deg')]).toInt16();
Export.table.toDrive({
  collection: statics.sampleRegions({collection: pts, properties: ['wc', 'lon', 'lat'], scale: 30, tileScale: 8}),
  description: 'gapfill_static', folder: FOLDER, fileFormat: 'CSV'});

// ---------------- monthly stacks ----------------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(REGION).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70));
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(REGION).filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');

function stackBands(list) {
  return ee.Image(ee.List(list).iterate(function (cur, acc) { return ee.Image(acc).addBands(ee.Image(cur)); }, ee.Image([])));
}
function monthStack(y, m) {
  var start = ee.Date.fromYMD(y, m, 1), end = start.advance(1, 'month');
  var tag = ee.String('_').cat(ee.Number(y).format('%d')).cat('_').cat(ee.Number(m).format('%02d'));
  var s2m = s2.filterDate(start, end).map(function (im) {
    var scl = im.select('SCL'); var ok = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
    return im.normalizedDifference(['B8', 'B4']).updateMask(ok).rename('ndvi');
  });
  var ndvi = s2m.median().multiply(1000).rename(ee.String('ndvi').cat(tag));
  var ncl = s2m.count().rename(ee.String('nobs').cat(tag));           // number of clear S2 observations that month
  var s1m = s1.filterDate(start, end).select(['VV', 'VH']);
  var vv = s1m.select('VV').mean().multiply(100).rename(ee.String('vv').cat(tag));      // dB x 100
  var vh = s1m.select('VH').mean().multiply(100).rename(ee.String('vh').cat(tag));
  var rain = chirps.filterDate(start, end).sum().rename(ee.String('rain').cat(tag)); // mm
  return ee.Image.cat([ndvi, ncl, vv, vh, rain]);
}
for (var y = Y0; y <= Y1; y++) {
  var months = ee.List.sequence(1, 12);
  var stack = stackBands(months.map(function (m) { return monthStack(y, m); }));
  stack = stack.unmask(-32768).toInt16();
  Export.table.toDrive({
    collection: stack.sampleRegions({collection: pts, properties: ['lon', 'lat'], scale: SCALE, tileScale: 16}),
    description: 'gapfill_' + y, folder: FOLDER, fileFormat: 'CSV'});
}
