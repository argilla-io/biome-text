import Vue from 'vue';
import Router from 'vue-router';
import ClassifierExplore from './views/ClassifierExplore';
import ErrorPage from './views/ErrorPage';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '*',
      redirect: 'error-page',
    },
    {
      path: '/error-page/',
      name: 'error-page',
      component: ErrorPage,
    },
    {
      path: '/:prediction/',
      name: 'explore',
      component: ClassifierExplore,
      props: true,
    },
  ],
});
