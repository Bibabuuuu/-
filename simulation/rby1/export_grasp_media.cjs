// Export the existing replay as a fully offline document and capture its actual UI.
const {chromium}=require('C:/Users/14737/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),http=require('node:http'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const dest=path.resolve('outputs/rby1-research/stage23');
const original=fs.readFileSync(path.join(dest,'replay_qa_preview.html'),'utf8');
const server=http.createServer((req,res)=>{res.setHeader('Content-Type','text/html; charset=utf-8');res.end(original);});
let browser;
(async()=>{
  try {
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    browser=await chromium.launch({channel:'msedge',headless:true});
    const context=await browser.newContext({viewport:{width:780,height:1100},deviceScaleFactor:1,colorScheme:'light'});
    const page=await context.newPage(),errors=[],responses=new Map();page.on('pageerror',e=>errors.push(e.message));
    page.on('response',response=>{if(/^https:\/\/(cdn\.jsdelivr\.net|unpkg\.com)\//.test(response.url()))responses.set(response.url(),response.body().then(body=>body.toString('utf8')));});
    await page.goto(`http://127.0.0.1:${server.address().port}/`,{waitUntil:'networkidle'});
    const frame=page.frameLocator('iframe');await frame.locator('[data-time]').filter({hasText:'4.351'}).waitFor();
    let inner=await page.locator('iframe').getAttribute('srcdoc');
    const urls=[...inner.matchAll(/<script[^>]+src="([^"]+)"/g)].map(match=>match[1]),scripts=new Map();
    for(const url of urls) {
      assert(/^https:\/\/(cdn\.jsdelivr\.net|unpkg\.com)\//.test(url));
      if(responses.has(url))scripts.set(url,await responses.get(url));
      else {
        const asset=await context.newPage();const response=await asset.goto(url,{waitUntil:'domcontentloaded'});
        assert(response?.ok(),url);scripts.set(url,await asset.locator('body').innerText());await asset.close();
      }
    }
    inner=inner.replace(/<script\b([^>]*?)\bsrc="([^"]+)"([^>]*?)>\s*<\/script>/g,(_,a,url,b)=>{
      assert(scripts.has(url),url);return `<script${a}${b}>\n${scripts.get(url).replace(/<\/script/gi,'<\\/script')}\n</script>`;
    });
    assert(!/<script[^>]+src=/.test(inner));
    const attr=inner.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;');
    const offline=original.replace(/data-srcdoc="[^"]*"/,()=>`data-srcdoc="${attr}"`);
    const output=path.join(dest,'grasp-stop-replay-offline.html');fs.writeFileSync(output,offline);
    const offlineContext=await browser.newContext({viewport:{width:780,height:1100},deviceScaleFactor:1,colorScheme:'light',offline:true});
    const offlinePage=await offlineContext.newPage(),network=[];offlinePage.on('pageerror',e=>errors.push(e.message));
    offlinePage.on('request',req=>{if(/^https?:/.test(req.url()))network.push(req.url());});
    await offlinePage.goto(pathToFileURL(output).href,{waitUntil:'load'});
    const root=offlinePage.frameLocator('iframe').locator('#grasp-stop-replay');
    await root.locator('[data-time]').filter({hasText:'4.351'}).waitFor();
    const frames=path.join(dest,'media_frames');fs.mkdirSync(frames,{recursive:true});const times=[];
    for(let i=0;i<=350;i+=5) {
      await root.locator('[data-time-slider]').evaluate((el,index)=>{el.value=String(index);el.dispatchEvent(new Event('input',{bubbles:true}));},i);
      await root.locator('[data-frame]').evaluate(img=>img.decode());
      times.push(Number(await root.locator('[data-time]').textContent()));
      await root.screenshot({path:path.join(frames,`frame_${String(times.length-1).padStart(3,'0')}.png`)});
    }
    assert.equal(times.length,71);assert.equal(times.at(-1),6);assert.equal(network.length,0);assert.deepEqual(errors,[]);
    await root.locator('[data-play]').click();await offlinePage.waitForTimeout(220);await root.locator('[data-play]').click();
    assert(Number(await root.locator('[data-time]').textContent())>4.351);
    const result={offline_file:output,offline_bytes:fs.statSync(output).size,inlined_script_urls:urls,
      offline_file_open_and_controls_passed:true,external_network_requests_during_offline_test:network,
      screenshot_count:times.length,recorded_times_s:times,errors,
      note:'Standalone export of the existing saved-state replay; no model or controller changes.'};
    fs.writeFileSync(path.join(dest,'media_export_check.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({offline:output,frames:times.length,bytes:result.offline_bytes,offline_pass:true}));
  } finally {if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
