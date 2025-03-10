<?php
require_once '../db.php';





@session_start();
	unset($_SESSION['is_login']);
	unset($_SESSION['id']);
	unset($_SESSION['username']);
?>
<html>
	<head>
		<meta charset="utf-8">
		<title> Final project </title>
		<link rel="icon" type="image" href="head.png">
		<link href="final.css" rel="stylesheet" />
		<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"/>
	</head>

	 <body>
      
        <!-- Navigation-->
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="final.php">YAPPI Shop</a>
                <div class="search">
                <form action="final-search.php" method="post">
                            <input class="search-bar" type="text" name="keyword" id="search" placeholder="輸入名稱" style="width: 660px;">
                            <input type="submit" value="搜尋">
                            <!--<button class="search-btn">
                              <i class="fas fa-search"></i>
                            </button>-->
                            
                          
                          </form>
                          </div>
                <button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>
                <div class="collapse navbar-collapse" id="navbarResponsive">
                    <ul class="navbar-nav ml-auto">
                        <!--<li class="nav-item"><a class="nav-link" href="../cart/cart-insert.php" style="width:60px;">購物車</a></li>-->
                        <!--<li class="nav-item"><a class="nav-link" href="../cart/shelf-edit.php" style="width:60px;">貨架</a></li>-->
                        <li class="nav-item"><a class="nav-link" href="https://yappi.speedtestcustom.com" target="_blank" style="width: 80px;">網路太慢？</a></li>
                        
                    </ul>
                </div>
            </div>
        </nav>
        <!-- Page header-->
        <header>
		<div class="container">
		<div class="row">
		<div class="hi">
        <!-- Page header-->
        <header>
		<div class="container">
		<div class="row">
		<div class="hi">
            <div class="carousel slide" id="carouselExampleIndicators" data-ride="carousel">
                <ol class="carousel-indicators">
                    <li class="active" data-target="#carouselExampleIndicators" data-slide-to="0"></li>
                    <li data-target="#carouselExampleIndicators" data-slide-to="1"></li>
                    <li data-target="#carouselExampleIndicators" data-slide-to="2"></li>
                </ol>
                <div class="carousel-inner">
                    <div class="carousel-item active">
                        <img class="d-block w-100" src="https://www.brandinlabs.com/wp-content/uploads/2018/05/5959-1527661968.jpg" alt="..." />
                      
                    </div>
                    <div class="carousel-item">
                        <img class="d-block w-100" src="https://www.brandinlabs.com/wp-content/uploads/2018/05/12926-1526977647.jpg" alt="..." />
                        
                    </div>
                    <div class="carousel-item">
                        <img class="d-block w-100 "src="https://pic36.photophoto.cn/20150728/0847085776639472_b.jpg" alt="..." />
                        
                    </div>
                </div>
                <a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">
                    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                    <span class="sr-only">Previous</span>
                </a>
                <a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">
                    <span class="carousel-control-next-icon" aria-hidden="true"></span>
                    <span class="sr-only">Next</span>
                </a>
            </div>
			</div>
			<div class="hihi">
			<div class ="login">
			<div class="wrapper">
				<form
				id="form"
				class="form-signin"
				method=""  
				action="" 
				>       
				<h3 class="form-signin-heading">Login</h3>
				<input
					id="username"
					name="username" 
					type="text" 
					class="form-control" 
					placeholder="Username" 
					required=""

				>
				<input
					id="password"
					name="password"  
					type="password" 
					class="form-control" 
					placeholder="Password" 
					required=""
				> 
				
				<button 
					class="btn btn-lg btn-primary btn-block"  
					name="submit" 
					value="Login" 
					type="submit"
					 
				><b>登入</b></button>  
				<input type="button" class="btn btn-lg btn-primary btn-block" value="註冊" onclick="location.href='reg.php'" >
			
				</form>	
				
			</div>
			</div>
			</div>
			</div>
        </header>
        <!-- Page Content-->
        <!--<section class="py-5">
            <div class="container">
                <h1 class="mb-4">Welcome to Modern Business</h1>
                <!-- Marketing Icons Section
                <div class="row">
                    <div class="col-lg-4 mb-4 mb-lg-0">
                        <div class="card h-100">
                            <h4 class="card-header">Card Title</h4>
                            <div class="card-body"><p class="card-text">Lorem ipsum dolor sit amet, consectetur adipisicing elit. Sapiente esse necessitatibus neque.</p></div>
                            <div class="card-footer"><a class="btn btn-primary" href="#!">Learn More</a></div>
                        </div>
                    </div>
                    <div class="col-lg-4 mb-4 mb-lg-0">
                        <div class="card h-100">
                            <h4 class="card-header">Card Title</h4>
                            <div class="card-body"><p class="card-text">Lorem ipsum dolor sit amet, consectetur adipisicing elit. Reiciendis ipsam eos, nam perspiciatis natus commodi similique totam consectetur praesentium molestiae atque exercitationem ut consequuntur, sed eveniet, magni nostrum sint fuga.</p></div>
                            <div class="card-footer"><a class="btn btn-primary" href="#!">Learn More</a></div>
                        </div>
                    </div>
                    <div class="col-lg-4">
                        <div class="card h-100">
                            <h4 class="card-header">Card Title</h4>
                            <div class="card-body"><p class="card-text">Lorem ipsum dolor sit amet, consectetur adipisicing elit. Sapiente esse necessitatibus neque.</p></div>
                            <div class="card-footer"><a class="btn btn-primary" href="#!">Learn More</a></div>
                        </div>
                    </div>
					
                </div>
            </div>
        </section>-->
        <hr class="my-0" />
        <!-- Portfolio Section-->
        <section class="py-5 bg-light">
            <div class="container">
                <h2 class="mb-4">商品列表</h2>
					<div class="commodity"> 
						<div class="row1">
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

				<!--<html xmlns="http://www.w3.org/1999/xhtml">


				<meta http-equiv="Content-Type" content="text/html; charset=big5">
				<title>:::::Product Catelog:::::</title>
				-->
				<?php
				$pdo=new PDO('mysql:host=localhost;dbname=final;charset=utf8','root','');    //連結資料庫
                $sql=$pdo->prepare('select * from commodity where c_name like ?');    //執行Query查詢 
                $sql->execute(['%'.$_REQUEST['keyword'].'%']);
                foreach($sql->fetchAll() as $row){
				?>
				
									<div class="col-lg-4 col-sm-6 mb-4">
										<div class="card h-100">
										<div class="im"><?php echo '<p><img src="',$row['url1'],'"></p>';?></div>       <!--建議:250X300px照片-->
											<!--<a href="#!"><img class="card-img-top" src="$url" alt="..." /></a>-->
											<div class="card-body">
												<h4 class="card-title"><a href="commodity.php?c_id=<?php echo $row['c_id']; ?>" ><?php echo $row['c_name']; ?></a></h4>
												<p class="card-text">商品價格: <?php echo $row['price']; ?> <br> 商品存貨: <?php echo $row['quantity']; ?><br></p>
											</div>
										</div>
									</div>   
								
				<?php
					} //while
				?>
                
				
                <!--PAGE-->
                <!--<ul class="pagination justify-content-center">
                    <li class="page-item">
                        <a class="page-link" href="#!" aria-label="Previous">
                            <span aria-hidden="true">«</span>
                            <span class="sr-only">Previous</span>
                        </a>
                    </li>
                    <li class="page-item"><a class="page-link" href="#!">1</a></li>
                    <li class="page-item"><a class="page-link" href="#!">2</a></li>
                    <li class="page-item"><a class="page-link" href="#!">3</a></li>
                    <li class="page-item">
                        <a class="page-link" href="#!" aria-label="Next">
                            <span aria-hidden="true">»</span>
                            <span class="sr-only">Next</span>
                        </a>
                    </li>
                </ul>-->
				</div>
				</div>
<?php
if($total>$per_page){
	if($page!=1){
		echo '<a href="final.php?page='.($page-1);
		echo '">上一頁</a>';
	}else{
		echo '<span class="noLink">上一頁</span>';
	}
	echo ' ';
	for($pi=1;$pi<=$pages;$pi++){
		//echo $pi;
		if($pi==$page){
			echo ' [ '.$pi.' ] ';
		}else{
			echo '<a href="final.php?page='.($pi);
			echo '"> '.$pi.' </a>';
		}
	}
	echo ' ';
	if($page!=$pages){
		echo '<a href="final.php?page='.($page+1);
		echo '">下一頁</a>';
	}else{
		echo '<span class="noLink">下一頁</span>';
	}

}?>
				
            </div>
        </section>
        <hr class="my-0" />
        <!-- Features Section-->
        <section class="py-5">
            <div class="container">
                <div class="row">
                    <div class="col-lg-6">
                        <h2 class="mb-4">關於我們</h2>
                        <p>以下是本組優秀的成員們:</p>
                        <ul>
                            <li><strong><a href="https://www.facebook.com/max.luo.9" target="_blank">羅名志</a>&nbsp信箱 : max230620089@gmail.com</strong></li>
                            <li><strong><a href="https://www.facebook.com/profile.php?id=100003103436903" target="_blank">黃宇生</a>&nbsp信箱 : yusheng.tem08@nctu.edu.tw</strong></li>
                            <li><strong><a href="https://www.facebook.com/profile.php?id=100002156402915" target="_blank">張恩瑜</a>&nbsp信箱 : rachel891028.tem08@nctu.edu.tw</strong></li>
                            <li><strong><a href="https://www.facebook.com/genie.juo" target="_blank">郭盈均</a>&nbsp信箱 : genie0813205.tem08@nctu.edu.tw</strong></li>
                            <li><strong><a href="https://www.facebook.com/max.smith.9862/" target="_blank">劉冠銘</a>&nbsp信箱 : aqdenny2001@gmail.com</strong></li>
                        </ul>
                        <p>關於這次的小組期末作業，我們想做一個富有我們味道的購物網站，很棒吧！</p>
                    </div>
                    <div class="col-lg-6"><img class="img-fluid rounded" src="https://pic.pimg.tw/doctorlin/1510828181-435260026_n.jpg" alt="..." /></div>
                </div>
            </div>
        </section>
        <hr class="my-0" />
        <!-- Call to Action-->
        <aside class="py-5 bg-light">
            <div class="container">
                <div class="row">
                    
                    <div class="col-md-4"><a class="btn btn-lg btn-secondary btn-block" href="#!" style="width:1120px;">購物車</a></div>
                </div>
            </div>
        </aside>
        <!-- Footer-->
        <!--<footer class="py-5 bg-dark">
            <div class="container"><p class="m-0 text-center text-white">&copy; 2021 National Chiao Tung University</p>
			<p class="m-0 text-center text-white">Database Management-Final Project</p>
			<p class="m-0 text-center text-white">Produced by Group 7</p>
			</div>
        </footer>-->
        <!-- Bootstrap core JS-->
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js"></script>
        <!-- Core theme JS-->
        <script src="js/scripts.js"></script>
    </body>
		
		<footer>
			
		</footer>
		

</html>
<script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
    <script>
      //當文件準備好時，
      $(document).on("ready", function() {
				//當表單 sumbit 出去的時候
				$("form.form-signin").on("submit", function(){
				  //使用 ajax 送出 帳密給 verify_user.php
					$.ajax({
            type : "POST",
            url : "../verify_user.php", 
            data : {
              un : $("#username").val(), //使用者帳號
              pw : $("#password").val() //使用者密碼
            },
            dataType : 'html' //設定該網頁回應的會是 html 格式
          }).done(function(data) {
            
			console.log(data);
            if(data == "yes")
            {
 
              window.location.href = "final2.php"; 
            }
            else
            {
              alert("登入失敗，請確認帳號密碼");
            }
            
          }).fail(function(jqXHR, textStatus, errorThrown) {
          
            alert("有錯誤產生，請看 console log");
            console.log(jqXHR.responseText);
          });
	       
          return false;
				});
      });
    </script>
