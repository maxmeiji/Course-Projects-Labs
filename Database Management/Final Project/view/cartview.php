
<?php

require_once '../db.php';



@session_start();
if(!isset($_SESSION['is_login']) || !$_SESSION['is_login'])
{
	//直接轉跳到 index.php
	header("Location: /final/list.php");
}
require_once '../login_status.php';
?>
<?php



?>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
        <meta name="description" content="" />
        <meta name="author" content="" />
        <title>購物車</title>
        <link rel="icon" type="image/x-icon" href="assets/favicon.ico" />
        <script src="https://use.fontawesome.com/releases/v5.15.3/js/all.js" crossorigin="anonymous"></script>
        <link href="final.css" rel="stylesheet" />
    </head>
    <body>
        
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="final2.php">YAPPI Shop</a>
                <button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>
                <div class="collapse navbar-collapse" id="navbarResponsive">
                    <ul class="navbar-nav ml-auto">
                        <li class="nav-item"><a class="nav-link" href="about.html">結帳去</a></li>
                        <li class="nav-item"><a class="nav-link" href="final2.php">繼續逛</a></li>
                        <li class="nav-item"><a class="nav-link" href="contact.html">聊天室</a></li>
                        
                    </ul>
                </div>
            </div>
        </nav>
        
        <section class="py-5">
            <div class="container">
           
                <h1>
                    您的購物車
                    <small>購購購~</small>
                </h1>
                <ol class="breadcrumb mb-4">
                    <li class="breadcrumb-item"><a href="final2.php">首頁</a></li>
                    <li class="breadcrumb-item active">購物車</li>
                </ol>
                <div class="row">
                    
                        <?php
				$i=0;
				$where_str='';
				$per_page = 9; //每頁最多9筆，這數字可以自由更改
				if(isset($_GET['page'])){
					$page=$_GET['page'];
				}	else{
					$page=1;
				}
				$start = ($page-1)*$per_page;
				$where_str='';
				?>

                <?php

				$link = mysqli_connect($host,$dbuser,$dbpw,$dbname) or die ("-1");
				  mysqli_select_db($link,$dbname) or die("cannot connect to db");
				  $query = "SELECT * FROM commodity $where_str ORDER BY c_id DESC ";
				  $result = mysqli_query($link,$query) or die("cannot connect to table" . mysqli_error( ));
				  $I=0;
				  while ( $row = mysqli_fetch_array($result) ) {
					$I++;
				  }
				  $total=$I;
				  $pages=ceil($total/$per_page);
				  if($page>$pages)
				  {
					  $page=$pages;
				  }
				  $start = ($page-1)*$per_page;
				?>

                <?php
				$imgBaseUrl='testProdImg/'; //假設圖片放在testProdImg資料夾
				if($total>0){
				$orderBy='c_id';
				$upDown='desc';
				$i = $start;
				$query = "SELECT * FROM commodity $where_str 	ORDER BY $orderBy $upDown limit $start, $per_page";
				$result = mysqli_query($link,$query) or die("Error:" . mysqli_error( )); 
                $rs = mysqli_fetch_array($result);
                @$id=$_REQUEST['c_id'];
                if (!isset($_SESSION['commodity'])) {
	                $_SESSION['commodity']=[];
                }
                $count=0;
                if (isset($_SESSION['commodity'][$id])) {
	                $count=$_SESSION['commodity'][$id]['count'];
                }
                @$_SESSION['commodity'][$id]=[
	                @'c_name'=>$_REQUEST['c_name'], 
	                @'price'=>$_REQUEST['price'], 
	                @'count'=>$count+$_REQUEST['count']
                ];

				
                     if (!empty($_SESSION['commodity'])) {
                        $total=0;
                        foreach ($_SESSION['commodity'] as $id=>$commodity) { 
				?>

                            <div class="im"><?php echo '<p><img src="',$rs['url1'],'"></p>';?></div>  <!--圖片大小待調-->
                            <div class="col-lg-3 col-md-4 col-sm-6 mb-4">
                            <div class="card h-100">
                            <div class="card-body">
                                <h4 class="card-title"><a href="commodity.php?c_id=<?php echo $id; ?>" ><?php echo $commodity['c_name']; ?></a></h4>
                                <p class="card-text">商品價格: <?php echo $commodity['price']; ?> <br> 商品數量: <?php echo $commodity['count']; ?><br>小計: <?php $subtotal=$commodity['price']*$commodity['count']; echo $subtotal; ?> </p>
                                <?php $total+=$subtotal; ?>
                                <p class="card-text"><a href="../cart/cart-delete.php?id=<?php echo $id;?>">刪除</a></p>
                                
                            </div>
                        </div>
                    </div>
                    <?php
					} //if !empty
				    ?>
                <?php 
				} /* if $total>0 end */ 
				?>
                <?php 
				} //foreach
				?>
                 <section class="py-5">
            </div>
            <div class="container">
           
                <h1>
                合計:<?php echo $total; ?>元
                    
                </h1>
                  
                </div>
                
       
        
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js"></script>
       
        <script src="js/scripts.js"></script>
    </body>
</html>
