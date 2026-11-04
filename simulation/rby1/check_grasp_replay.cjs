// Black-box check of this task's read-only replay, in a fresh headless browser.
const {chromium}=require('C:/Users/14737/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),http=require('node:http'),assert=require('node:assert/strict');
const path=require('node:path');
const dest=path.resolve('outputs/rby1-research/stage23'),html=fs.readFileSync(path.join(dest,'replay_qa_preview.html'));
const server=http.createServer((req,res)=>{res.setHeader('Content-Type','text/html; charset=utf-8');res.end(html);});
const errors=[];let browser;
(async()=>{
  try {
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    browser=await chromium.launch({channel:'msedge',headless:true});
    const page=await browser.newPage({viewport:{width:780,height:1000},deviceScaleFactor:1});
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`,{waitUntil:'networkidle'});
    const frame=page.frameLocator('iframe'),root=frame.locator('#grasp-stop-replay');
    await root.locator('[data-time]').filter({hasText:'4.351'}).waitFor();
    assert.equal(await root.locator('[data-chart] path[stroke-width="1.8"]').count(),4);
    const initial=await root.locator('[data-frame]').getAttribute('src');
    await root.locator('[data-time-slider]').evaluate(el=>{el.value=el.max;el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));});
    assert.equal(await root.locator('[data-time]').textContent(),'6.000');
    assert.match(await root.locator('[data-relative]').textContent(),/0.376/);
    assert.notEqual(await root.locator('[data-frame]').getAttribute('src'),initial);
    const hover=root.locator('[data-chart-hover-overlay]'),box=await hover.boundingBox();await page.mouse.move(box.x+box.width*.6,box.y+box.height*.7);
    assert.equal(await root.locator('[data-tip]').isVisible(),true);
    assert.match(await root.locator('[data-tip]').textContent(),/上下 Δz/);
    await root.locator('[data-legend] button').nth(3).click();
    assert.equal(await root.locator('[data-chart] path[stroke-width="1.8"]').count(),3);
    await root.locator('[data-legend] button').nth(3).click();
    await root.locator('[data-play]').click();await page.waitForTimeout(250);await root.locator('[data-play]').click();
    const playedTime=Number(await root.locator('[data-time]').textContent());assert(playedTime>4.351 && playedTime<6);
    const geometry=[];
    for(const width of [780,360,320]) {
      await page.setViewportSize({width,height:1100});
      const measure=await root.evaluate(el=>{const r=el.getBoundingClientRect();const svgs=[...el.querySelectorAll('[data-local],[data-chart]')];return {width:r.width,scrollWidth:document.documentElement.scrollWidth,viewport:window.innerWidth,
        clippedSvgText:svgs.flatMap(svg=>[...svg.querySelectorAll('text')].filter(t=>{const b=t.getBoundingClientRect(),s=svg.getBoundingClientRect();return b.width>0 && (b.left<s.left-.5 || b.right>s.right+.5 || b.top<s.top-.5 || b.bottom>s.bottom+.5);}).map(t=>t.textContent))};});
      assert(measure.scrollWidth<=measure.viewport+1,JSON.stringify(measure));assert.deepEqual(measure.clippedSvgText,[]);geometry.push({outer_width:width,...measure});
      await page.screenshot({path:path.join(dest,`replay_qa_${width}.png`),fullPage:true});
    }
    await page.emulateMedia({colorScheme:'dark'});await page.screenshot({path:path.join(dest,'replay_qa_dark.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    const result={passed:true,slider_updates_time_values_image_and_local_motion:true,series_toggle_and_hover:true,play_pause_advances_without_loop:true,errors,geometry,
      source:'Saved Stage-23 middle trajectory only; fresh browser, no user profile.',headless_launch_requires_host_process:true};
    fs.writeFileSync(path.join(dest,'replay_ui_checks.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
  } finally {if(browser)await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
