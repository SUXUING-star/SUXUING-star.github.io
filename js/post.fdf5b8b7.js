"use strict";(self["webpackChunkxingyunchaju"]=self["webpackChunkxingyunchaju"]||[]).push([[205],{7725:function(e,t,n){n.r(t),n.d(t,{default:function(){return R}});var s=n(6768),l=n(4232);const a={key:0,class:"detail-post"},i={class:"detail-header"},o={class:"detail-meta"},r=["datetime"],c={class:"detail-word-count"},d={class:"detail-reading-time"},h={class:"detail-container"},u={class:"detail-main"},p=["innerHTML"],g={class:"detail-sidebar"},v={key:0,class:"sidebar-section toc-section"},w=["innerHTML"],k={class:"sidebar-section progress-section"},m={class:"progress-bar"},f={class:"progress-text"},L={class:"sidebar-section recent-posts"},x={class:"recent-posts-list"},y={class:"post-date"};function $(e,t,n,$,b,C){const E=(0,s.g2)("router-link");return $.post?((0,s.uX)(),(0,s.CE)("div",a,[(0,s.Lk)("div",i,[(0,s.Lk)("h1",null,(0,l.v_)($.post.title),1),(0,s.Lk)("div",o,[(0,s.Lk)("time",{datetime:$.post.date},(0,l.v_)($.post.date),9,r),t[0]||(t[0]=(0,s.Lk)("div",{class:"meta-divider"},null,-1)),(0,s.Lk)("span",c,(0,l.v_)($.wordCount)+" 字",1),t[1]||(t[1]=(0,s.Lk)("div",{class:"meta-divider"},null,-1)),(0,s.Lk)("span",d,(0,l.v_)($.readingTime)+" 分钟阅读",1)])]),(0,s.Lk)("div",h,[(0,s.Lk)("div",u,[(0,s.Lk)("div",{class:"detail-content markdown-body",innerHTML:$.renderedContent,ref:"contentRef"},null,8,p)]),(0,s.Lk)("aside",g,[$.tableOfContents?((0,s.uX)(),(0,s.CE)("div",v,[t[2]||(t[2]=(0,s.Fv)('<h3 class="sidebar-title"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="21" y1="10" x2="7" y2="10"></line><line x1="21" y1="6" x2="3" y2="6"></line><line x1="21" y1="14" x2="3" y2="14"></line><line x1="21" y1="18" x2="7" y2="18"></line></svg> 目录 </h3>',1)),(0,s.Lk)("nav",{class:"toc-nav",innerHTML:$.tableOfContents},null,8,w)])):(0,s.Q3)("",!0),(0,s.Lk)("div",k,[t[3]||(t[3]=(0,s.Lk)("h3",{class:"sidebar-title"},[(0,s.Lk)("svg",{xmlns:"http://www.w3.org/2000/svg",width:"16",height:"16",viewBox:"0 0 24 24",fill:"none",stroke:"currentColor","stroke-width":"2","stroke-linecap":"round","stroke-linejoin":"round"},[(0,s.Lk)("path",{d:"M12 20h9"}),(0,s.Lk)("path",{d:"M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"})]),(0,s.eW)(" 阅读进度 ")],-1)),(0,s.Lk)("div",m,[(0,s.Lk)("div",{class:"progress-inner",style:(0,l.Tr)({width:$.readingProgress+"%"})},null,4)]),(0,s.Lk)("div",f,(0,l.v_)($.readingProgress)+"%",1)]),(0,s.Lk)("div",L,[t[4]||(t[4]=(0,s.Fv)('<h3 class="sidebar-title"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg> 最近文章 </h3>',1)),(0,s.Lk)("ul",x,[((0,s.uX)(!0),(0,s.CE)(s.FK,null,(0,s.pI)($.recentPosts,(e=>((0,s.uX)(),(0,s.CE)("li",{key:e.id,class:(0,l.C4)({active:e.id===$.post.id})},[(0,s.bF)(E,{to:"/post/"+e.id},{default:(0,s.k6)((()=>[(0,s.eW)((0,l.v_)(e.title)+" ",1),(0,s.Lk)("span",y,(0,l.v_)(e.date),1)])),_:2},1032,["to"])],2)))),128))])])])])])):(0,s.Q3)("",!0)}n(4114),n(8992),n(2577),n(3949);var b=n(144),C=n(1387),E=n(4249),M=n(357),T=n(1017),j=n.n(T),H=(n(2093),n(2006),n(3206),n(5686),n(895),n(2059),n(9399),n(7316),{name:"PostDetailPage",setup(){const e=(0,C.lq)(),t=(0,E.Pj)(),n=(0,b.KR)(null),l=(0,b.KR)(0),a=(0,b.KR)(""),i=(0,s.EW)((()=>t.state.posts.find((t=>t.id===parseInt(e.params.id))))),o=(0,s.EW)((()=>i.value?.content?i.value.content.replace(/\s+/g,"").length:0)),r=(0,s.EW)((()=>Math.ceil(o.value/300))),c=(0,s.EW)((()=>t.state.posts.slice().sort(((e,t)=>new Date(t.date)-new Date(e.date))).slice(0,5))),d=()=>{window.MathJax={tex:{inlineMath:[["$","$"],["\\(","\\)"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},svg:{fontCache:"global"},options:{skipHtmlTags:["script","noscript","style","textarea","pre"]}};const e=document.createElement("script");e.src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js",e.async=!0,document.head.appendChild(e)},h=()=>{const e=new M.xI.Renderer;e.heading=(e,t)=>{const n=e.replace(/<[^>]*>/g,""),s=n.toLowerCase().replace(/[\s\u4e00-\u9fa5]+/g,"-").replace(/[^\w\u4e00-\u9fa5-]/g,"").replace(/-+/g,"-").replace(/^-|-$/g,"");return`\n          <h${t} id="${s}" class="heading-anchor">\n            <span class="anchor-marker" id="${s}-marker"></span>\n            ${e}\n          </h${t}>\n        `};const t=e.paragraph.bind(e);e.paragraph=e=>{const n=/\$([^$]+)\$/g;if(e=e.replace(n,((e,t)=>`<span class="inline-math">\\(${t}\\)</span>`)),e.startsWith("$$")&&e.endsWith("$$")){const t=e.slice(2,-2);return`<div class="math-block">\\[${t}\\]</div>`}return t(e)},M.xI.setOptions({renderer:e,gfm:!0,breaks:!0,smartLists:!0})},u=()=>{if(!n.value)return console.warn("contentRef is null"),"";const e=n.value.querySelectorAll("h2, h3, h4");if(console.log("Headings found:",e.length),0===e.length)return console.warn("No headings found in the content"),"";let t='<nav class="table-of-contents"><ul class="toc-list">',s=2,l=[s];e.forEach(((e,n)=>{console.log(`Heading ${n}:`,{text:e.textContent,tagName:e.tagName,id:e.id});const a=parseInt(e.tagName.charAt(1)),i=e.textContent.trim(),o=e.id||`heading-${n}`;if(a>s)t+='<ul class="toc-sublist">',l.push(a);else if(a<s)while(l.length>0&&l[l.length-1]>=a)t+="</ul>",l.pop();t+=`\n      <li class="toc-item level-${a}">\n        <a href="#${o}" class="toc-link" data-level="${a}">\n          ${i}\n        </a>\n      </li>\n    `,s=a}));while(l.length>1)t+="</ul>",l.pop();return t+="</ul></nav>",console.log("Generated TOC:",t),t},p=e=>e?M.xI.parse(e):"",g=(0,s.EW)((()=>{if(!i.value?.content)return"";try{return p(i.value.content)}catch(e){return console.error("Markdown rendering error:",e),"<p>Error rendering content</p>"}})),v=()=>{window.MathJax&&window.MathJax.typesetPromise?.()},w=e=>{const t=e.target.closest(".toc-link");if(t){e.preventDefault();const n=t.getAttribute("href").slice(1),s=document.getElementById(n);if(s){const e=80,t=s.getBoundingClientRect().top,n=t+window.pageYOffset-e;window.scrollTo({top:n,behavior:"smooth"}),s.classList.add("highlight"),setTimeout((()=>{s.classList.remove("highlight")}),2e3)}}};(0,s.wB)(g,(()=>{(0,s.dY)((()=>{j().highlightAll(),a.value=u(),k(),v();const e=document.querySelector(".table-of-contents");e&&e.addEventListener("click",w)}))})),(0,s.sV)((()=>{h(),d(),window.addEventListener("scroll",k),(0,s.dY)((()=>{j().highlightAll(),a.value=u(),k()}))})),(0,s.xo)((()=>{window.removeEventListener("scroll",k);const e=document.querySelector(".table-of-contents");e&&e.removeEventListener("click",w)}));const k=()=>{if(!n.value)return;const e=n.value.offsetHeight,t=window.scrollY,s=window.innerHeight,a=t/(e-s)*100;l.value=Math.min(Math.max(Math.round(a),0),100)};return{post:i,contentRef:n,tableOfContents:a,readingProgress:l,wordCount:o,readingTime:r,recentPosts:c,renderedContent:g}}}),P=n(1241);const W=(0,P.A)(H,[["render",$]]);var R=W}}]);
//# sourceMappingURL=post.fdf5b8b7.js.map