/******/ (() => { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ "./label_studio/static/dm/js/main.js":
/*!*******************************************!*\
  !*** ./label_studio/static/dm/js/main.js ***!
  \*******************************************/
/***/ ((__unused_webpack_module, exports) => {

/*! For license information please see main.js.LICENSE.txt */


/***/ }),

/***/ "./label_studio/static/js/modules/index.js":
/*!*************************************************!*\
  !*** ./label_studio/static/js/modules/index.js ***!
  \*************************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony import */ var _dm_js_main__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../../dm/js/main */ "./label_studio/static/dm/js/main.js");
// import '@heartex/datamanager/build/static/css/main.css';


const dmRoot = document.querySelector(".datamanager");

if (dmRoot) {
  const dm = new _dm_js_main__WEBPACK_IMPORTED_MODULE_0__.DataManager({
    root: dmRoot,
    apiGateway: "../api",

  });

  console.log(dm);
}


/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		if(__webpack_module_cache__[moduleId]) {
/******/ 			return __webpack_module_cache__[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	(() => {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = (exports) => {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	})();
/******/ 	
/************************************************************************/
/******/ 	// startup
/******/ 	// Load entry module
/******/ 	__webpack_require__("./label_studio/static/js/modules/index.js");
/******/ 	// This entry module used 'exports' so it can't be inlined
/******/ })()
;
//# sourceMappingURL=index.js.map