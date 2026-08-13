// SPDX-License-Identifier: MPL-2.0
"use strict";
const DEMO_SCHEMA="qec.panel-browser-demonstration.v1";
const digitsInput=document.querySelector("#digits"), eventsList=document.querySelector("#events"), recordBox=document.querySelector("#record");
const selectors={"bank-a":document.querySelector("#selector-a"),"bank-b":document.querySelector("#selector-b")};
let currentRecord=null;
function makeRecord(){
  const digits=digitsInput.value.split(",").map(value=>Number.parseInt(value.trim(),10));
  if(!digits.length||digits.some(value=>!Number.isInteger(value)||value<0)) throw new Error("Digits must be non-negative integers.");
  return {schema:DEMO_SCHEMA,contract_version:"171.5",browser_demo_only:true,canonical_receipt:false,evidence_class:"demonstration",digits,selected_path:"path-a",events:[
    {sequence:0,phase:"register",action:"digit_register_sealed"},{sequence:1,phase:"control",action:"sender_program_sealed"},
    {sequence:2,phase:"actuation",action:"selector_move",bank:"bank-a",selector_position:4},{sequence:3,phase:"actuation",action:"path_connected",path_id:"path-a"},
    {sequence:4,phase:"verification",action:"independent_route_verification"},{sequence:5,phase:"commit",action:"commit_verified_route"}]};
}
function render(record){eventsList.replaceChildren();document.querySelectorAll(".bank").forEach(node=>node.classList.remove("active"));Object.values(selectors).forEach(node=>{node.style.transform="translateY(0px)";});for(const event of record.events){const item=document.createElement("li");item.textContent=`${event.sequence} · ${event.phase} · ${event.action}`;eventsList.append(item);if(event.action==="selector_move"){selectors[event.bank].style.transform=`translateY(${event.selector_position*20}px)`;document.querySelector(`#${event.bank}`).classList.add("active");}}recordBox.textContent=JSON.stringify(record,null,2);}
function sonify(record){const AudioContextClass=window.AudioContext||window.webkitAudioContext;if(!AudioContextClass)throw new Error("WebAudio unavailable.");const context=new AudioContextClass();record.events.forEach((event,index)=>{const oscillator=context.createOscillator(),gain=context.createGain();oscillator.frequency.value=220+index*55;gain.gain.value=.06;oscillator.connect(gain);gain.connect(context.destination);const start=context.currentTime+index*.12;oscillator.start(start);oscillator.stop(start+.08);});}
document.querySelector("#run").addEventListener("click",()=>{try{currentRecord=makeRecord();render(currentRecord);}catch(error){recordBox.textContent=error.message;}});
document.querySelector("#sound").addEventListener("click",()=>{if(!currentRecord)currentRecord=makeRecord();sonify(currentRecord);});
currentRecord=makeRecord();render(currentRecord);
