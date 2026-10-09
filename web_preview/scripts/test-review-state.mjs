import test from "node:test";
import assert from "node:assert/strict";
import {isAccepted,deriveStatistics,refreshOrganization} from "../src/review-state.js";

test("manual decisions override old publication and statistics",()=>{
  const e={category:"定位球",subtype:"corner",publishedForStatistics:true};
  assert.equal(isAccepted({...e,reviewStatus:"rejected"}),false);
  assert.equal(isAccepted({...e,publishedForStatistics:false,reviewStatus:"confirmed"}),true);
  const layer={unavailable:[]};
  assert.equal(deriveStatistics(layer,[e]).reviewedMetrics.acceptedCount,1);
  const stats=deriveStatistics(layer,[{...e,reviewStatus:"rejected"}]);
  assert.equal(stats.reviewedMetrics.acceptedCount,0);
  assert.deepEqual(stats.reviewedMetrics.subtypes,{});
  assert.equal(stats.counts[0].rejected,1);
  assert.equal(isAccepted({}),false);
});

test("organization loses confirmed status after source rejection or cut",()=>{
  const layer={items:[{sourceEventIds:["a","b"],publishedForStatistics:true}]};
  const events=[{id:"a",publishedForStatistics:true,sceneId:0},{id:"b",publishedForStatistics:true,sceneId:0}];
  assert.equal(refreshOrganization(layer,events).items[0].publishedForStatistics,true);
  assert.equal(refreshOrganization(layer,[events[0],{...events[1],reviewStatus:"rejected"}]).items[0].publishedForStatistics,false);
  assert.equal(refreshOrganization(layer,[events[0],{...events[1],sceneId:2}]).items[0].publishedForStatistics,false);
  assert.equal(layer.items[0].publishedForStatistics,true);
});
