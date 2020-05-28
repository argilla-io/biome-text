import Vue from 'vue';
import Router from 'vue-router';
import ClassifierExplore from './views/ClassifierExplore';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      redirect: '/:prediction/',
    },
    {
      path: '/:prediction/',
      name: 'explore',
      component: ClassifierExplore,
      props: true,
    },
  ],
});
