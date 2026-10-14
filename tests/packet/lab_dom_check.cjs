// SPDX-License-Identifier: MPL-2.0
// DOM smoke checks only; this is not a rendered-browser or acoustic assessment.
'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'../../docs/labs/packet/index.html'),'utf8');
const bundled=html.match(/<script id="cases" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element{
  constructor(){this.textContent='';this.value='';this.listeners={};this.attrs={};this.children=[];this.files=[];}
  setAttribute(k,v){this.attrs[k]=v;}
  append(e){this.children.push(e);}
  replaceChildren(){this.children=[];}
  addEventListener(k,v){this.listeners[k]=v;}
  click(){}
}
const ids=new Map(),document={getElementById(id){if(!ids.has(id))ids.set(id,new Element());return ids.get(id);},createElement(){return new Element();},createElementNS(){return new Element();}};
document.getElementById('cases').textContent=bundled;
let audioStarts=0,exportedFiles=[];
class AudioContext{
  constructor(){this.currentTime=0;this.destination={};}
  async resume(){}
  createOscillator(){return {frequency:{value:0},connect(){},start(){audioStarts++;},stop(){}};}
  createGain(){return {gain:{setValueAtTime(){},exponentialRampToValueAtTime(){}},connect(){}};}
}
const sandbox={document,window:{AudioContext},Blob:class {constructor(parts){exportedFiles.push(parts.join(''));}},URL:{createObjectURL(){return 'blob:offline-test';},revokeObjectURL(){}},setTimeout(fn){fn();},console};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
const get=id=>document.getElementById(id);
(async()=>{
  assert(get('position').textContent.startsWith('Event 1 /'));
  for(const scenario of ['success','failover','overflow','corrupt','drop','recovery']){
    get('scenario').value=scenario;get('scenario').listeners.change();get('next').listeners.click();
    assert(get('position').textContent.startsWith('Event 2 /'));
    assert(get('topology').children.length>0);
    assert(get('results').children.length>0);
    get('export').listeners.click();
    const exported=JSON.parse(exportedFiles.at(-1));
    assert.equal(exported.schema,'qec.packet-demo.v1');assert.equal(exported.cryptographic_proof,false);
    get('file').files=[{size:exportedFiles.at(-1).length,async text(){return exportedFiles.at(-1);}}];await get('file').listeners.change();
    assert(get('status').textContent.startsWith('Loaded a demonstration'));
  }
  assert.equal(audioStarts,0);await get('sound').listeners.click();get('next').listeners.click();assert.equal(audioStarts,1);
  await get('sound').listeners.click();get('next').listeners.click();assert.equal(audioStarts,1);
  for(const invalid of [{schema:'qec.packet-fabric-receipt.v1'}, {...JSON.parse(exportedFiles.at(-1)),cryptographic_proof:true}, {...JSON.parse(exportedFiles.at(-1)),results:[null]}]){
    get('file').files=[{size:100,async text(){return JSON.stringify(invalid);}}];await get('file').listeners.change();assert(get('status').textContent.startsWith('Could not load'));
  }
  const empty={...JSON.parse(exportedFiles.at(-1)),events:[],results:[]};get('file').files=[{size:100,async text(){return JSON.stringify(empty);}}];await get('file').listeners.change();get('next').listeners.click();assert.equal(get('position').textContent,'Event 0 / 0');
  console.log('Packet lab DOM checks passed: six cases, steps, exports/imports, audio opt-in, malformed and empty inputs.');
})().catch(error=>{console.error(error);process.exitCode=1;});
